"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
import numpy as np
import torch
import json
import torch.backends.cudnn as cudnn
from MovieChat.common.config import Config
from MovieChat.common.dist_utils import get_rank
from MovieChat.common.registry import registry
from MovieChat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle
import decord
import cv2
import time
from tqdm import tqdm
import subprocess
from moviepy.editor import VideoFileClip
from decord import VideoReader
decord.bridge.set_bridge('torch')

# imports modules for registration
from MovieChat.datasets.builders import *
from MovieChat.models import *
from MovieChat.processors import *
from MovieChat.runners import *
from MovieChat.tasks import *
from moviepy.editor import*

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import GPUtil

MAX_INT = 8
N_SAMPLES = 32
SHORT_MEMORY_Length = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--video-path", required=True, help="path to video file.")
    parser.add_argument("--gt_file", required=True, help="path to gt file.")
    parser.add_argument("--gt_file_answers", required=True, help="path to gt file.")
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--fragment-video-path", required=True, help="path to video fragment file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config_seed):
    seed = config_seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def video_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)
 
def capture_video(video_path, fragment_video_path, per_video_length, n_stage):
    start_time = n_stage * per_video_length
    end_time = (n_stage+1) * per_video_length
    video =CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])
    video.write_videofile(fragment_video_path)

    
def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def parse_video_fragment(video_path, video_length, n_stage = 0, n_samples = N_SAMPLES):
    decord.bridge.set_bridge("torch")
    per_video_length = video_length / n_samples
    # cut video from per_video_length(n_stage-1, n_stage)
    capture_video(video_path, fragment_video_path, per_video_length, n_stage)
    return fragment_video_path

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_context_emb(self, input_text, msg, img_list):
        
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers in detail.###Human: <Video><ImageHere></Video>"
        prompt_2 = input_text
        prompt_3 = "###Assistant:"

        prompt = prompt_1 + " " + prompt_2 + prompt_3

        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def answer(self, img_list, input_text, msg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
            repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs = self.get_context_emb(input_text, msg, img_list) 

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p, 
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty, 
            temperature=temperature, 
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text, output_token.cpu().numpy()
    
    def cal_frame(self, video_length, cur_min, cur_sec, middle_video):
        per_frag_second = video_length / N_SAMPLES
        if middle_video:
            cur_seconds = cur_min * 60 + cur_sec
            num_frames = int(cur_seconds / per_frag_second)
            per_frame_second = per_frag_second/SHORT_MEMORY_Length
            cur_frame = int((cur_seconds-per_frag_second*num_frames)/per_frame_second)
            return num_frames, cur_frame
        else:
            cur_frame = 0
            num_frames = int(video_length / per_frag_second)
            return num_frames, cur_frame

    def upload_video_without_audio(self, video_path, fragment_video_path, cur_min, cur_sec, cur_image, img_list, middle_video):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            video_length = video_duration(video_path) 
            num_frames, cur_frame = self.cal_frame(video_length, cur_min, cur_sec, middle_video)
            if num_frames == 0:
                video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=0, n_samples= N_SAMPLES)
                video_fragment, msg = load_video(
                    video_path=fragment_video_path,
                    n_frms=MAX_INT, 
                    height=224,
                    width=224,
                    sampling ="uniform", return_msg = True
                ) 
                video_fragment = self.vis_processor.transform(video_fragment)
                video_fragment = video_fragment.unsqueeze(0).to(self.device)

                self.model.encode_short_memory_frame(video_fragment, cur_frame)
            else:
                for i in range(num_frames):
                    print(i)
                    video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=i, n_samples= N_SAMPLES)
                    video_fragment, msg = load_video(
                        video_path=fragment_video_path,
                        n_frms=MAX_INT, 
                        height=224,
                        width=224,
                        sampling ="uniform", return_msg = True
                    )
                    video_fragment = self.vis_processor.transform(video_fragment) 
                    video_fragment = video_fragment.unsqueeze(0).to(self.device)

                    if middle_video:
                        self.model.encode_short_memory_frame(video_fragment, cur_frame)
                    else:
                        self.model.encode_short_memory_frame(video_fragment)
                
        else:
            raise NotImplementedError
        video_emb, _ = self.model.encode_long_video(cur_image, middle_video)
        img_list.append(video_emb) 
        return msg  


if __name__ =='__main__':
    config_seed = 42
    setup_seeds(config_seed)
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_qa = json.load(file)
    
    with open(args.gt_file_answers) as file2:
        gt_answers = json.load(file2)
    
    output_list = []
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    index = 0
    for sample in tqdm(gt_qa):
        chat.model.long_memory_buffer = []
        chat.model.short_memory_buffer = []
        
        video_id = sample['video_name']
        question = sample['question']
        answer = gt_answers[index]['answer']
        index = index + 1
        # answer = sample['answer']
        id = sample['question_id']
        sample_set = {'id': id, 'question': question, 'answer': answer}
        video_path = os.path.join(args.video_path, f"{video_id}.mp4")

        fragment_video_path = args.fragment_video_path
        cur_min = 0
        cur_sec = 0
        middle_video = False

        cap = cv2.VideoCapture(video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        cur_fps = fps_video * (60*cur_min + cur_sec)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_fps)
        ret, frame = cap.read()
        temp_frame_path = 'src/output_frame/snapshot.jpg'

        cv2.imwrite(temp_frame_path, frame) 
        raw_image = Image.open(temp_frame_path).convert('RGB') 
        image = chat.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(chat.device) # [1,3,1,224,224]
        cur_image = chat.model.encode_image(image)

        img_list = []
        msg = chat.upload_video_without_audio(
            video_path=video_path, 
            fragment_video_path=fragment_video_path,
            cur_min=cur_min, 
            cur_sec=cur_sec, 
            cur_image = cur_image, 
            img_list=img_list, 
            middle_video = middle_video,
            )
    
        text_input = question
        num_beams = args.num_beams
        temperature = args.temperature
        
        try:
            llm_message = chat.answer(img_list=img_list,
                                input_text=text_input,
                                msg = msg,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
            sample_set['pred'] = llm_message
            print(llm_message)
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_id}': {e}")
    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)
