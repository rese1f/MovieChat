import logging
import random

from PIL import Image

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from MovieChat.common.registry import registry
from MovieChat.models.blip2 import Blip2Base, disabled_train
from MovieChat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BertConfig
import einops
import copy
from MovieChat.models.Qformer import BertConfig, BertLMHeadModel

import queue
import numpy as np
import math
from scipy.spatial.distance import cosine

from skimage import transform
import cv2
import clip

@registry.register_model("moviechat")
class MovieChat(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/moviechat.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  
        device_8bit=0, 
        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        short_memory_length = 18,
        long_memory_length = 256,
        short_memory_merge = 2,
        Qformer_input = 8,
        n_position = 16,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Frame Filter')
        device = 'cuda:0'
        self.filter_model, self.filter_preprocess = clip.load("ViT-B/32", device=device)
        print('Loading Frame Filter Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = model.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.max_frame_pos = max_frame_pos
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size) #[32,768] [200]

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None


        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        self.Qformer_input = Qformer_input
        logging.info('create short-memory buffer')
        self.short_memory_length = short_memory_length 
        self.short_memory_buffer = []
        self.short_memory_merge = short_memory_merge 
        self.temp_short_memory = []

        logging.info('create long-memory buffer')
        self.long_memory_length = long_memory_length 
        self.long_memory_buffer = []

        logging.info('whether Question the whole video')
        self.middle_video =False
        self.question_minute = None
        self.question_second = None

        # expand position embedding
        self.n_position = n_position


        # calculate the position_embedding
        self.frame_position_embeddings = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_short_memory_frame(self, videofragment, question, n_frame:int = 16):
        device = videofragment.device
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = videofragment.size() # batch_size:1 time_length:8
        videofragment = einops.rearrange(videofragment, 'b c t h w -> (b t) c h w') 
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(videofragment)).to(device) 
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # filter frames
            tokenize_text = clip.tokenize(question).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = self.filter_model(videofragment, tokenize_text)
                probs = logits_per_text.softmax(dim=-1).cpu().numpy()
            max_sim = np.mean(probs)


            if max_sim < 0.25:
                self.short_memory_merge = 1
            else:
                self.short_memory_merge = 3


            # load short_memory_buffer
            cur_frame = 0
            q_hidden_state = query_output.last_hidden_state 
            for frame in q_hidden_state:
                if cur_frame < n_frame:
                    if len(self.short_memory_buffer) == self.short_memory_length:
                        self.short_memory_buffer.pop(0)
                    self.short_memory_buffer.append(frame)
                cur_frame += 1

            self.temp_short_memory = []
            for i in self.short_memory_buffer:
                self.temp_short_memory.append(i)
            
            #merge short_memory_frames
            similar_list = []
            for frame_i in range(len(self.short_memory_buffer) -1):
                scores = self.short_memory_buffer[frame_i] @ self.short_memory_buffer[frame_i+1].transpose(-1, -2)
                token_scores = scores.diag()
                frame_silimar = torch.mean(scores)
                similar_list.append(frame_silimar)
            
            while len(self.short_memory_buffer) > self.short_memory_merge:
                max_value = max(similar_list)
                max_index = similar_list.index(max_value)
                new_frame_feature = (self.short_memory_buffer[max_index].cpu()+self.short_memory_buffer[max_index+1].cpu())/2
                self.short_memory_buffer[max_index] = new_frame_feature.cuda()
                del(self.short_memory_buffer[max_index+1])
                similar_list = []
                for frame_i in range(len(self.short_memory_buffer)-1):
                    scores = self.short_memory_buffer[frame_i] @ self.short_memory_buffer[frame_i+1].transpose(-1, -2)
                    frame_silimar = torch.mean(scores)
                    similar_list.append(frame_silimar)
            for frame in self.short_memory_buffer:
                self.long_memory_buffer.append(frame)
            self.short_memory_buffer = []

    def encode_long_video(self, cur_image, middle_video:False):
        
        device = 'cuda:0'
        # input shape b,c,t,h,w
        batch_size = 1 # batch_size:1 
        self.long_memory_buffer = [i.unsqueeze(0) for i in self.long_memory_buffer]
        self.n_position = math.ceil(math.sqrt(len(self.long_memory_buffer)))
        if self.n_position >= 32:
            self.n_position = 32
        position_ids = torch.arange(self.n_position).long().to(self.query_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) 
        p = self.video_frame_position_embedding(position_ids).squeeze(0)
         
        self.u = []
        self.frame_position_embeddings = []
        self.alpha = 0.01 
        for p_i in p:
            u_i = (p_i-self.alpha * p[0])/(1-self.alpha)
            self.u.append(u_i)
        for i in range(self.n_position):
            for j in range(self.n_position):
                q_i = self.alpha * self.u[i] + (1-self.alpha) * self.u[j] 
                q_i = q_i.unsqueeze(0)
                self.frame_position_embeddings.append(q_i)
        self.frame_position_embeddings = torch.cat(self.frame_position_embeddings, dim = 0)

        if middle_video:
            while (len(self.long_memory_buffer)+len(self.temp_short_memory)+1) > self.frame_position_embeddings.shape[0]:
                if len(self.temp_short_memory) != 0:
                    self.temp_short_memory.pop(0)
                else:
                    self.long_memory_buffer.pop(0)
            
            if len(self.long_memory_buffer) == 0:
                self.temp_short_memory = [i.unsqueeze(0) for i in self.temp_short_memory]
                cur_short = torch.cat(self.temp_short_memory, dim = 0)
                video_features = torch.cat([cur_short], dim = 0)
            else:
                cur_video = torch.cat(self.long_memory_buffer,dim = 0)
                self.temp_short_memory = [i.unsqueeze(0) for i in self.temp_short_memory]
                if len(self.temp_short_memory) != 0:
                    cur_short = torch.cat(self.temp_short_memory, dim = 0)
                    video_features = torch.cat([cur_video,cur_short], dim = 0)
                else:
                    video_features = torch.cat([cur_video], dim = 0)
                video_features = torch.cat([video_features, cur_image], dim = 0)
            
            cur_video = []
            cur_pos = []
            for i in range(len(video_features)):
                    cur_pos.append(self.frame_position_embeddings[i])
                    cur_video.append(video_features[i])
            
            cur_pos = [j.unsqueeze(0) for j in cur_pos]
            cur_video = [j.unsqueeze(0) for j in cur_video]
            cur_position_embeddings = torch.cat(cur_pos, dim=0)
            cur_position_embeddings = cur_position_embeddings.unsqueeze(-2) 
            cur_position_embeddings = cur_position_embeddings.unsqueeze(0)
            frame_hidden_state = torch.cat(cur_video, dim=0)
            frame_hidden_state = einops.rearrange(frame_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=len(video_features))
                
            frame_hidden_state = cur_position_embeddings.to(device) + frame_hidden_state.to(device)
                
            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=len(video_features)) 
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1) 
            # a video Q-former to aggregate frame-level representations 
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, 
                encoder_hidden_states=frame_hidden_state, 
                encoder_attention_mask=frame_atts, 
                return_dict=True,
            )
            video_hiddens=video_query_output.last_hidden_state 

        # a linear layer to project the output video representations into the same dimension as the text embeddings of LLMs
            inputs_llama = self.llama_proj(video_hiddens)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
            return inputs_llama, atts_llama

        else:   
            while len(self.long_memory_buffer) > self.frame_position_embeddings.shape[0]:
                    self.long_memory_buffer.pop(0)        
            cur_video = []
            cur_pos = []
            for i in range(len(self.long_memory_buffer)):
                cur_pos.append(self.frame_position_embeddings[i])
                cur_video.append(self.long_memory_buffer[i])
            cur_pos = [j.unsqueeze(0) for j in cur_pos]
            cur_position_embeddings = torch.cat(cur_pos, dim=0)
            cur_position_embeddings = cur_position_embeddings.unsqueeze(-2) 
            cur_position_embeddings = cur_position_embeddings.unsqueeze(0)
            frame_hidden_state = torch.cat(cur_video, dim=0) #[1,32,768]
            frame_hidden_state = einops.rearrange(frame_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=len(self.long_memory_buffer)) #[64,32,768]
                
            frame_hidden_state = cur_position_embeddings.to(device) + frame_hidden_state.to(device)
                
            # frame attention
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=len(self.long_memory_buffer)) 
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device) 
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1) 
            # a video Q-former to aggregate frame-level representations 
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            video_hiddens=video_query_output.last_hidden_state 



        # a linear layer to project the output video representations into the same dimension as the text embeddings of LLMs
            inputs_llama = self.llama_proj(video_hiddens)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device) 
            return inputs_llama, atts_llama

    def encode_image(self, image):
        device = 'cuda:0'

        image = einops.rearrange(image, 'b c t h w -> (b t) c h w') 
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) 
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            q_hidden_state = query_output.last_hidden_state

        return q_hidden_state



    def encode_videoQformer_visual(self, image):
        device = image.device
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size() # batch_size:1 time_length:8
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w') 
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) 
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            q_hidden_state = query_output.last_hidden_state 

            # merge after every frame added
            for frame in q_hidden_state:
                self.long_memory_buffer.append(frame)
            similar_list = []
            for frame_i in range(self.long_memory_length):
                similar_list.append(cosine(self.long_memory_buffer[frame_i].flatten().cpu(), self.long_memory_buffer[frame_i+1].flatten().cpu()))
            while len(self.long_memory_buffer) > self.long_memory_length:
                max_value = max(similar_list)
                max_index = similar_list.index(max_value)
                new_frame_feature = (self.long_memory_buffer[max_index].cpu()+self.long_memory_buffer[max_index+1].cpu())/2
                self.long_memory_buffer[max_index] = new_frame_feature.cuda()
                del(self.long_memory_buffer[max_index+1])
                similar_list = []
                for frame_i in range(len(self.long_memory_buffer)-1):
                    similar_list.append(cosine(self.long_memory_buffer[frame_i].flatten().cpu(), self.long_memory_buffer[frame_i+1].flatten().cpu()))
            
            #  a position embedding layer to inject temporal information into video frames
            if self.whole_video:
                # add frame_pos embedding
                self.long_memory_buffer = [i.unsqueeze(0) for i in self.long_memory_buffer]
                for i in self.long_memory_buffer:
                    while len(i.shape) > 3:
                        i = i.squeeze(0)
                frame_hidden_state = torch.cat(self.long_memory_buffer,dim = 0)
                position_ids = torch.arange(self.long_memory_length, dtype=torch.long, device=query_tokens.device) 
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
                frame_hidden_state = einops.rearrange(frame_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=self.long_memory_length)
                frame_hidden_state = frame_position_embeddings + frame_hidden_state
                
                # frame attention
                frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=self.long_memory_length)
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            
            # a video Q-former to aggregate frame-level representations 
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state
            # a linear layer to project the output video representations into the same dimension as the text embeddings of LLMs
            inputs_llama = self.llama_proj(video_hidden) 
            
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            num_patch_tokens = self.num_video_query_token
            img_embeds, atts_img = self.encode_videoQformer_visual(image)
               
            temp_input_ids = copy.deepcopy(input_ids) # just copy input_ids
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            img_embeds, atts_img = self.encode_videoQformer_visual(image)

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            num_video_query_token=num_video_query_token,
        )
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
