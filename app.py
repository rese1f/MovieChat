import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from MovieChat.common.config import Config
from MovieChat.common.dist_utils import get_rank
from MovieChat.common.registry import registry
from MovieChat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle
import decord
import cv2
import time
import subprocess
from moviepy.editor import VideoFileClip
from decord import VideoReader
import gradio as gr

import pandas as pd
import plotly.express as px
from helpers import *
decord.bridge.set_bridge('torch')

#%%

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

#%%
def parse_args():
    import pdb;pdb.set_trace()
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
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
    import pdb;pdb.set_trace()
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


# ========================================
#             Model Initialization
# ========================================
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

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    import pdb;pdb.set_trace()
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

     
def gradio_ask(user_message, chatbot, chat_state):
    import pdb;pdb.set_trace()
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    import pdb;pdb.set_trace()
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    import pdb;pdb.set_trace()
    llm_message = chat.answer(img_list=img_list,
                              input_text=user_message,
                              msg=msg,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list
def video_duration(filename):
    import pdb;pdb.set_trace()
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)
 
def capture_video(video_path, fragment_video_path, per_video_length, n_stage):
    import pdb;pdb.set_trace()
    start_time = n_stage * per_video_length
    end_time = (n_stage+1) * per_video_length
    video =CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])
    video.write_videofile(fragment_video_path)

    
def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    import pdb;pdb.set_trace()
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
    # print(type(temp_frms))
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
    import pdb;pdb.set_trace()
    decord.bridge.set_bridge("torch")
    per_video_length = video_length / n_samples
    # cut video from per_video_length(n_stage-1, n_stage)
    fragment_video_path = "src/video_fragment/output.mp4"
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

    def upload_imgorvideo(self, gr_video, text_input, chat_state,chatbot, breakpoint,minute:int=None, second:int=None):
        import pdb;pdb.set_trace()
        if breakpoint[0] == "Breakpoint mode":
            chat.model.middle_video = True
            chat.model.question_minute = minute
            chat.model.question_second = second
        else:
            chat.model.middle_video = False

        cap = cv2.VideoCapture(gr_video)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        cur_fps = fps_video * (60*minute + second)

        cap = cv2.VideoCapture(gr_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_fps)
        ret, frame = cap.read()
        temp_frame_path = 'output_frame/snapshot.jpg'
        cv2.imwrite(temp_frame_path, frame)
        raw_image = Image.open(temp_frame_path).convert('RGB') 
        image = chat.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(chat.device) # [1,3,1,224,224]
        cur_image = chat.model.encode_image(image)


        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state = default_conversation.copy()
        chat_state = Conversation(
            system= "You are able to understand the visual content that the user provides."
            "Follow the instructions carefully and explain your answers in detail.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
        img_list = [] 
        llm_message = self.upload_video_without_audio(gr_video, cur_image, img_list)
        # 在这里报错
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot

    
    def get_context_emb(self, input_text, msg, img_list):
        import pdb;pdb.set_trace()
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers in detail.###Human: <Video><ImageHere></Video>"
        # prompt_2 = msg
        prompt_3 = input_text
        prompt_4 = "###Assistant:"
        # prompt = prompt_1 + " " + prompt_2 + "  " + prompt_3 + prompt_4

        prompt = prompt_1 + " " + prompt_3 + prompt_4

        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # import pdb;pdb.set_trace()

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def answer(self, img_list, input_text, msg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
            repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        import pdb;pdb.set_trace()
        
        embs = self.get_context_emb(input_text, msg, img_list) # embs = [1,142,4096],  img.shape = [1,32,4096]

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        
        # import pdb;pdb.set_trace()
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,# 1
            top_p=top_p, # 0.9
            repetition_penalty=repetition_penalty, # 1.0
            length_penalty=length_penalty, # 1
            temperature=temperature, # 1
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
        import pdb;pdb.set_trace()
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

    def upload_video_without_audio(self, video_path, cur_image, img_list):
        import pdb;pdb.set_trace()
        msg = ""
        fragment_video_path = "src/video_fragment/output.mp4"
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            video_length = video_duration(video_path) 

            # num_frames = N+SAMPLES
            # import pdb;pdb.set_trace()
            num_frames, cur_frame = self.cal_frame(video_length, self.model.question_minute, self.model.question_second, self.model.middle_video)


            if num_frames == 0:
                video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=0, n_samples= N_SAMPLES)
                video_fragment, msg = load_video(
                    video_path=fragment_video_path,
                    n_frms=MAX_INT, # here!!!!!,change the time_length, origin:8
                    height=224,
                    width=224,
                    sampling ="uniform", return_msg = True
                ) # video.shape [3,8,224,224]
                video_fragment = self.vis_processor.transform(video_fragment) # [3,8,224,224]
                video_fragment = video_fragment.unsqueeze(0).to(self.device)


                self.model.encode_short_memory_frame(video_fragment, cur_frame)
            else:
                for i in range(num_frames):
                    print(i)
                    video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=i, n_samples= N_SAMPLES)
                    video_fragment, msg = load_video(
                        video_path=fragment_video_path,
                        n_frms=MAX_INT, # here!!!!!,change the time_length, origin:8
                        height=224,
                        width=224,
                        sampling ="uniform", return_msg = True
                    ) # video.shape [3,8,224,224]
                    video_fragment = self.vis_processor.transform(video_fragment) # [3,8,224,224]
                    video_fragment = video_fragment.unsqueeze(0).to(self.device)

                    # import pdb;pdb.set_trace()
                    if self.model.middle_video:
                        self.model.encode_short_memory_frame(video_fragment, cur_frame)
                    else:
                        self.model.encode_short_memory_frame(video_fragment)
                
        else:
            raise NotImplementedError
        # import pdb;pdb.set_trace()
        # video_emb, _ = self.model.encode_long_video()  
        video_emb, _ = self.model.encode_long_video(cur_image, self.model.middle_video)
        # append the video embedding after videoQformer
        img_list.append(video_emb) # 1
        return msg 

LIBRARIES = ["Breakpoint mode", "Global mode"]
def create_pip_plot(libraries, pip_choices):
    import pdb;pdb.set_trace()
    if "Pip" not in pip_choices:
        return gr.update(visible=False)
    output = retrieve_pip_installs(libraries, "Cumulated" in pip_choices)
    df = pd.DataFrame(output).melt(id_vars="day")
    plot = px.line(df, x="day", y="value", color="variable",
                   title="Pip installs")
    plot.update_layout(legend=dict(x=0.5, y=0.99),  title_x=0.5, legend_title_text="")
    return gr.update(value=plot, visible=True)


def create_star_plot(libraries, star_choices):
    import pdb;pdb.set_trace()
    if "Stars" not in star_choices:
        return gr.update(visible=False)
    output = retrieve_stars(libraries, "Week over Week" in star_choices)
    df = pd.DataFrame(output).melt(id_vars="day")
    plot = px.line(df, x="day", y="value", color="variable",
                   title="Number of stargazers")
    plot.update_layout(legend=dict(x=0.5, y=0.99),  title_x=0.5, legend_title_text="")
    return gr.update(value=plot, visible=True)


def create_issue_plot(libraries, issue_choices):
    import pdb;pdb.set_trace()
    if "Issue" not in issue_choices:
        return gr.update(visible=False)
    output = retrieve_issues(libraries,
                             exclude_org_members="Exclude org members" in issue_choices,
                             week_over_week="Week over Week" in issue_choices)
    df = pd.DataFrame(output).melt(id_vars="day")
    plot = px.line(df, x="day", y="value", color="variable",
                   title="Cumulated number of issues, PRs, and comments",
                   )
    plot.update_layout(legend=dict(x=0.5, y=0.99),  title_x=0.5, legend_title_text="")
    return gr.update(value=plot, visible=True)




title = """
<h1 align="center"><a href="https://rese1f.github.io/MovieChat"></a> </h1>

<h1 align="center">MovieChat: From Dense Token to Sparse Memory in Long Video Understanding</h1>

<h5 align="center">  Introduction:MovieChat, a novel framework that integrating vision models and LLMs, is the first to support long video understanding . </h5> 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://rese1f.github.io/MovieChat'><img src='https://img.shields.io/badge/Github-Code-success'></a>
<a href='https://rese1f.github.io/MovieChat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> 
<a href='https://rese1f.github.io/MovieChat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> 
<a href='https://rese1f.github.io/MovieChat'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>


Thank you for using the MovieChat Demo Page! If you have any questions or feedback, feel free to contact us. 

If you find MovieChat interesting, please give us a star on GitHub.

Current online demo uses the 7B version of MovieChat due to resource limitations. 


"""

Note_markdown = ("""
### Note
Video-LLaMA is a prototype model and may have limitations in understanding complex scenes, long videos, or specific domains.
The output results may be influenced by input quality, limitations of the dataset, and the model's susceptibility to illusions. Please interpret the results with caution.

**Copyright 2023 Alibaba DAMO Academy.**
""")

cite_markdown = ("""
## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023videollama,
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  year = 2023,
  journal = {arXiv preprint arXiv:2306.02858}
  url = {https://arxiv.org/abs/2306.02858}
}
""")

case_note_upload = ("""
### We provide some examples at the bottom of the page. Simply click on them to try them out directly.
""")

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            video = gr.Video()
            gr.Markdown(case_note_upload)

            gr.Markdown("## Select inference mode")
            libraries = gr.CheckboxGroup(choices=LIBRARIES, label="")

            with gr.Column(scale=0.5):
                minute = gr.Slider(
                minimum=0,
                maximum=20,
                value=1,
                step=1,
                interactive=True,
                label="minutes of breakpoint)",
                )

                second = gr.Slider(
                minimum=0,
                maximum=60,
                value=1,
                step=1,
                interactive=True,
                label="seconds of breakpoint)",
                )


            
            with gr.Column(scale=0.5):
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam search numbers)",
                )
                    
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
                clear = gr.Button("Restart")
            gr.Markdown(Note_markdown)
        
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='MovieChat')
            text_input = gr.Textbox(label='User', placeholder='Upload your image/video first, or directly click the examples at the bottom of the page.', interactive=False)
            

    with gr.Column():
        gr.Examples(examples=[
            [f"examples/225199575-1-208.mp4", "What are this couple doing? "],
            [f"examples/566079305-1-208.mp4", "Can you describe the video? "],
            [f"examples/Cooking_cake.mp4", "What is going on in the kitchen? "],
            [f"examples/pandas.mp4", "what are the pandas doing?"],
        ], inputs=[video, text_input])
        
    gr.Markdown(cite_markdown)
    
    chat = Chat(model, vis_processor, device='cuda:0')


    upload_button.click(chat.upload_imgorvideo, [video, text_input, chat_state, chatbot, libraries, minute, second])
   

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, text_input, upload_button, chat_state, img_list], queue=False)
    

demo.launch(share=False, enable_queue=True)


# %%