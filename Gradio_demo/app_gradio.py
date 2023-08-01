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


from MovieChat.datasets.builders import *
from MovieChat.models import *
from MovieChat.processors import *
from MovieChat.runners import *
from MovieChat.tasks import *
from moviepy.editor import*
from inference import *

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

def show_video(video):
    print(video)

# ========================================
#             Gradio Setting
# ========================================


LIBRARIES = ["Breakpoint mode", "Global mode"]

title = """
<h1 align="center"><a href="https://rese1f.github.io/MovieChat"></a> </h1>

<h1 align="center">MovieChat: From Dense Token to Sparse Memory in Long Video Understanding</h1>

<h5 align="center">  Introduction:MovieChat, a novel framework that integrating vision models and LLMs, is the first to support long video understanding . </h5> 


Thank you for using the MovieChat Demo Page! If you have any questions or feedback, feel free to contact us. 

If you find MovieChat interesting, please give us a star on GitHub.

Current online demo uses the 7B version of MovieChat due to resource limitations. 

Please note that after clicking the chat button, you will need to view the result in the terminal window.


"""

case_note_upload = ("""
### We provide some examples at the bottom of the page. Simply click on them to try them out directly.
""")

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)


    with gr.Column(scale=0.5):
        video = gr.Video()
        gr.Markdown(case_note_upload)
        
        with gr.Column():
            upload_button = gr.Button(value="Upload", interactive=True, variant="primary")
            chat_state = gr.State()
            img_list = gr.State()
            text_input = gr.Textbox(label='User', placeholder='Upload your image/video first, or directly click the examples at the bottom of the page.', interactive=True)
            gr.Markdown("## Select inference mode")
            libraries = gr.CheckboxGroup(choices=LIBRARIES, label="")

        with gr.Column(scale=0.5):
            with gr.Row():
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

            with gr.Row():
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
                
        
        with gr.Column():
            upload_text = gr.Button("Chat now")

    with gr.Column():
        gr.Examples(examples=[
            [f"src/examples/Cooking_cake.mp4", "What is going on in the kitchen? "],
            [f"src/examples/goblin.mp4", "Can you describe the movie?"],
        ], inputs=[video, text_input])
        
    upload_button.click(show_video, [video])

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

    
    upload_text.click(chat.gener_infer,[video, text_input, num_beams, temperature, libraries, minute, second])
    
demo.launch(share=False, enable_queue=True)




