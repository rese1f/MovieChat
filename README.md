<img src="src/assets/logo.png" height="120px" align="left">

# MovieChat

[![](http://img.shields.io/badge/cs.CV-arXiv%3A2307.16449-B31B1B.svg)](https://arxiv.org/abs/2307.16449v4)
[![](http://img.shields.io/badge/cs.CV-arXiv%3A2404.17176-B31B1B.svg)](https://arxiv.org/abs/2404.17176)

> **MovieChat: From Dense Token to Sparse Memory for Long Video Understanding**  
> Enxin Song*, Wenhao Chai*, Guanhong Wang*, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Xun Guo, Tian Ye, Yan Lu, Jenq-Neng Hwang, Gaoang Wang✉️   
> _CVPR 2024._


<img width="1155" alt="image" src="https://github.com/user-attachments/assets/4c0412d3-0729-4f56-af0c-1ee3eeac8f99">

MovieChat can handle videos with >10K frames on a 24GB graphics card. MovieChat has a 10000× advantage over other methods in terms of the average increase in GPU memory cost per frame (21.3KB/f to ~200MB/f).
<p align="center" width="100%">
<a target="_blank"><img src="src/assets/wave.gif" alt="MovieChat" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 🔢 MovieChat-1K leaderboard

Feel free to PR your new results!

| Model with Link | Comment | Breakpoint Acc | Global Acc |
|-----------------------------------------------|------------------------------|------------|----------------|
| [VILA1.5-8B](https://openreview.net/pdf?id=oS79Tw3G0c)     | Eng-to-end                  |  -   | 40.0 |
| [Video-LLaMA](https://arxiv.org/pdf/2306.02858)            | End-to-end                  | 39.1 | 51.7 |
| [VideoChat](https://arxiv.org/abs/2305.06355)              | End-to-end                  | 46.1 | 57.8 |
| [TimeChat](https://arxiv.org/pdf/2406.11333)               | CoT, ICL, train on MovieChat| 46.1 | 73.8 |
| [VideoChatGPT](https://arxiv.org/pdf/2306.05424)           | End-to-end                  | 48.0 | 47.6 |
| [MovieChat](https://arxiv.org/abs/2307.16449v4) (baseline) | End-to-end                  | 48.3 | 62.3 |
| [MovieChat+](https://arxiv.org/abs/2404.17176) (baseline)  | End-to-end                  | 49.6 | 71.2 |
| [Long-LLaVA](https://arxiv.org/abs/2411.13093)             | Eng-to-end                  | 54.0 | 69.6 |
| [Long-LLaVA + Video-RAG](https://arxiv.org/abs/2411.13093) | Eng-to-end                  | 54.5 | 72.9 |
| [Streaming Long Video](https://arxiv.org/abs/2405.16009)   | Train on MovieChat          | 54.9 | 90.4 |
| [HEM-LLM](https://arxiv.org/pdf/2409.06299)                | Unknown training dataset    | -    | 90.6 |
| [DrVideo](https://arxiv.org/pdf/2406.12846)                | RAG                         | 56.7 | 93.1 |
| [ReWind](https://arxiv.org/pdf/2411.15556)                 | End-to-end                  | 57.2 | 87.6 |
| [HERMES](https://arxiv.org/pdf/2408.17443)                 | Train on MovieChat          | 57.3 | 78.6 |
| [llavaonevision-MovieChat](https://github.com/rese1f/MovieChat) | End-to-end             | -    | 79.0 |
| [Flash-VStream](https://arxiv.org/abs/2406.08085)          | Train on MovieChat          | 59.6 | 96.0 |
| [MM-Screenplayer](https://arxiv.org/pdf/2406.17309)        | RAG                         | 68.8 | 87.5 |
| [Sullam Jeoung, _et al_](https://arxiv.org/pdf/2410.20252) | Agent                       | -    | 84.8 |


## 🔢 Evaluation of MovieChat on Existing Benchmarks

Sort in alphabetical order.

| Benchmark | Results |
|-----------|---------|
| ActivityNet-QA | Acc. / Score: 45.7 / 3.4 |
| Charades-STA | R@1(IOU =0.3): 8.8 • R@1(IOU =0.5): 2.9 •  R@1(IOU =0.7): 1.3 |
| CineClipQA | Overall: 20.86/2.11 • Description: 23.67/2.41 • Intention: 30.19/2.41 • Perception: 21.80/1.97 • Temporality: 16.32/1.97 • Spaciality: 16.40/1.98 |
| CVRR-ES | Average: 16.41 |
| EgoSchema | Top 1 Acc: 53.5 |
| EventBench | Acc: 20.33 |
| InfiniBench | Global Appearance: 6.59 • Scene transition: 6.41 • Character actions: 4.51 • Temporal order: 36.99 • Local visual: 17.76 • Summarization: 0.14 • Deep context: 0.55 • Spoiler questions: 0.34 • Multiple events: 0.85 • Avg: 14.45/0.47 |
| LvBench | ER: 21.3 • EU: 23.1 • KIR: 25.9 • TG: 22.3 • Rea: 24.0 • Sum: 17.2 • Overall: 22.5 |
| LvM-QA | Acc. / Score: 48.3 / 2.57 |
| MLVU | Holistic TR: 29.5 • AR: 25.0 • VS: 2.33 • Single Detail NQA: 24.2 • ER: 24.7 • PQA: 25.8 • SSC: 3.23 • Multi Detail AO: 28.6 • AC: 22.8 • M-Avg: 25.8 • G-Avg: 2.78 |
| MovieChat-1K | Global Acc. / Score: 62.3 / 3.23 • Global Acc. / Score: 48.3 / 2.57 |
| MovieCORE | Acc: 20.33 • Comp: 2.90 • Depth: 2.29 • Evid: 2.14 • Coh: 2.30 • Avg: 2.23 |
| MSVD-QA | Acc. / Score: 75.2 / 3.8 |
| MSRVTT-QA | Acc. / Score: 52.7 / 2.6 |
| MVBench | Avg: 55.1 |
| NExT-QA | Acc. / Score: 49.9 / 2.7 |
| QVHighlight | mAP: 11.7 • HIT @1: 16.1 |
| RVS-Ego | Acc. / Score: 50.7 / 3.4 |
| RVS-Movie | Acc. / Score: 36.0 / 2.3 |
| Seed-Bench | Procedure Understanding: 29.82 • Action Recognition: 40.11 |
| SFD | Multiple-Choice V: 8.4 • L: 16.4 • VL: 8.0 • Open-Ended V: 14.0 • L: 15.7 • VL: 11.8 |
| SVBench | Dialogue SA: 20.46 • Dialogue CC: 20.05 • Dialogue LC: 27.76 • Dialogue TU: 21.81 • Dialogue IC: 22.21 • Dialogue OS: 21.89 • Streaming SA: 17.99 • Streaming CC: 16.42 • Streaming LC: 20.37 • Streaming TU: 15.77 • Streaming IC: 19.08 • Streaming OS: 17.43 |
| TV-Caption | BertScore: 38.11 • CIDER: 8.43 • ROUGE-L: 12.09 • SPICE: 9.21 |
| VCG Bench | CI: 2.76 • DO: 2.93 • CU: 3.01 • TU: 2.24 • CO: 2.42 • Avg: 2.67 |
| VDC | Camera: 37.25/1.98 • Short: 32.55/1.59 • Background: 28.99/1.54 • Main: 31.97/1.64 • Object: 28.82/1.46 • Avg: 31.92/1.64 |
| VideoMME | w/o subs: 38.2 • w/o subs (Long): 33.4 |
| Video-ChatGPT | Avg: 2.67 • CI: 2.76 • DO: 2.93 • CU: 3.01 • TU: 2.24 • CO: 2.42 |
| VS-Ego | Acc. / Score: 52.2 / 3.4 |
| VS-Movie | Acc. / Score: 39.1 / 2.3 |
| YouCook2 | C: 38.5 • M: 18.8 |


## :fire: News
* **[2024.10.26]** :keyboard: We upload MovieChat, MovieChat_OneVision, MovieChat-1K to [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
* **[2024.10.26]** :keyboard: We release a new version of MovieChat, which use LLaVA-OneVision as the base model instead of the original VideoLLaMA. The new version is available on [MovieChat_Onevision](https://github.com/rese1f/MovieChat/tree/main/MovieChat_Onevision).
* **[2024.6.13]** :film_projector: We release the ground truth of MovieChat's test set in [Hugging Face](https://huggingface.co/datasets/Enxin/MovieChat-1K-test). 
* **[2024.5.10]** :film_projector: We release the raw videos of MovieChat's training set in [Hugging Face](https://huggingface.co/datasets/Enxin/MovieChat-1K_train). 
* **[2024.4.29]** :page_with_curl: We update the MovieChat+ [paper](https://arxiv.org/abs/2404.17176) with implementation details, technical evaluations, and dataset information.
* **[2024.4.25]** :keyboard:We update a new version of MovieChat+. We realse the [MovieChat+ code](https://github.com/rese1f/MovieChat/blob/main/MovieChat/models/moviechat%2B.py) and the corresponding [evaluation code](https://github.com/rese1f/MovieChat/blob/main/eval_code/result_prepare/run_inference_qa_moviechat%2B.py). Our paper is Coming soon!
* **[2024.4.19]** :keyboard:We update the latest source code of MovieChat to [PyPI](https://pypi.org/). Now you can use MovieChat by `pip install Moviechat` directly!
* **[2024.3.25]** :bar_chart: We host challenge track 1 of [the 4th International Workshop on Long-form Video Understanding: Towards Multimodal AI Assistant and Copilot](https://cvpr.thecvf.com/Conferences/2024/workshop-list) at CVPR 2024. You can participate in the challenge and submit your results via [Codalab](https://codalab.lisn.upsaclay.fr/competitions/18284?secret_key=bd5e312c-4775-43cf-933b-70726d00bcbe). We will display the results on the [leaderboard](https://espere-1119-song.github.io/LOVEU-CVPR-24-Track-1-Leaderboard/). For each participant, we hope you can submit your results in JSON format and report both the average running time and VRAM usage. We will use these metrics to select the most efficient method. For detailed information about the challenge, please refer to this [link](https://sites.google.com/view/loveucvpr24/track1).
* **[2024.3.11]** :film_projector: We release the test set of the MovieChat-1K in [Hugging Face](https://huggingface.co/datasets/Enxin/MovieChat-1K-test). Each video contains 3 global questions and 10 breakpoint questions.
* **[2024.2.27]** :tada: Our paper was accepted by CVPR 2024!
* **[2024.2.14]** :film_projector: We release the training set of the MovieChat-1K in [Hugging Face](https://huggingface.co/datasets/Enxin/MovieChat-1K_train). Due to copyright restrictions, we share the clip features extracted by [eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), containing 8192 frames of each video.
* **[2023.11.27]** :page_with_curl: We update the [paper](https://arxiv.org/pdf/2307.16449v2.pdf) with implementation details, technical evaluations, and dataset information.
* **[2023.11.23]** :keyboard:We update the latest source code of MovieChat.
* **[2023.8.1]** :page_with_curl: We release the [paper](https://arxiv.org/abs/2307.16449).
* **[2023.7.31]** :keyboard:We release eval [code and instraction](https://github.com/rese1f/MovieChat/tree/main/eval_code) for short video QA on **MSVD-QA**, **MSRVTT-QA** and **ActivityNet-QA**.
* **[2023.7.29]** :joystick:We release [Gradio demo](https://github.com/rese1f/MovieChat/tree/main/Gradio_demo) of MovieChat.
* **[2023.7.22]** :keyboard:We release source code of MovieChat.
  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/moviechat-from-dense-token-to-sparse-memory/zeroshot-video-question-answer-on-activitynet)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-activitynet?p=moviechat-from-dense-token-to-sparse-memory)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/moviechat-from-dense-token-to-sparse-memory/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=moviechat-from-dense-token-to-sparse-memory)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/moviechat-from-dense-token-to-sparse-memory/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=moviechat-from-dense-token-to-sparse-memory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/moviechat-from-dense-token-to-sparse-memory/zero-shot-long-video-global-mode-question)](https://paperswithcode.com/sota/zero-shot-long-video-global-mode-question?p=moviechat-from-dense-token-to-sparse-memory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/moviechat-from-dense-token-to-sparse-memory/zero-shot-long-video-breakpoint-mode-question)](https://paperswithcode.com/sota/zero-shot-long-video-breakpoint-mode-question?p=moviechat-from-dense-token-to-sparse-memory)

## 📊Performance Comparison on MovieChat-1K
| **Method**         | **Text Decoder**   | **# Frames** | **Global Mode Acc.** | **Global Mode Sco.** |
|--------------------|--------------------|--------------|----------------------|----------------------|
| GIT                | non-LLM based      | 6            | 28.8                 | 1.83                 |
| mPLUG-2            | non-LLM based      | 8            | 31.7                 | 2.13                 |
| **Video Chat**     | LLM based          | 32           | 57.8                 | 3.00                 |
| **Video LLaMA**    | LLM based          | 32           | 51.7                 | 2.67                 |
| **Video-ChatGPT**  | LLM based          | 100          | 47.6                 | 2.55                 |
| **MovieChat**      | LLM based          | 2048         | 62.3                 | 3.23                 |
| **MovieChat+**     | LLM based          | 2048         | 71.2                 | 3.51               |
| **MovieChat-Onevision**  | LLM based    | 2048         | **79.0**             | **4.20**             |

## ✨How to run MovieChat quickly?

We have packaged MovieChat and uploaded it to PyPI. To run MovieChat quickly, you need to install it firstly. 
```
pip install MovieChat
```
We advise you to install version `0.6.3` for now. Since `MovieChat` will download checkpoints from Huggingface automatically, if your service doesn't support `git clone from <HuggingFace  url>`, we recommend you to download the checkpoint to your service, and change the respective path in the package, including [q_former_model](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth), [ckpt_path](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-vicuna7b-v2.pth?download=true), and [llama_model](https://huggingface.co/Enxin/MovieChat-vicuna). 

Before you run the following inference code, we hope you can verify the installation of `ffprobe` via `ffprobe -version`. This command should return the version of ffprobe if it is correctly installed. Otherwise, you should install it via `sudo apt-get install ffmpeg` (Ubuntu).

```
from PIL import Image
import cv2

from MovieChat.processors.video_processor import AlproVideoEvalProcessor
from MovieChat.models.chat_model import Chat
from MovieChat.models.moviechat import MovieChat

device = 'cuda:0'
print('Initializing Chat')
moviechat_model = MovieChat.from_config(device=device).to(device)
vis_processor_cfg = {'name': 'alpro_video_eval', 'n_frms': 8, 'image_size': 224}
frame_processor = AlproVideoEvalProcessor.from_config(vis_processor_cfg)
chat = Chat(moviechat_model, frame_processor, device=device)
print('Initialization Finished')

video_path = "Your video path, end with mp4"
fragment_video_path = "The path to store tmp video clips"
middle_video = False # True->Breakpoint mode, False->Global mode
question = "Your Question"
cur_min = 0 # Change it when Breakpoint mode
cur_sec = 0 # Change it when Breakpoint mode

cap = cv2.VideoCapture(video_path)
cur_fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, cur_fps)
ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_frame)
image = chat.image_vis_processor(pil_image).unsqueeze(0).unsqueeze(2).half().to(device)
cur_image = chat.model.encode_image(image)

img_list = []
msg = chat.upload_video_without_audio(
    video_path=video_path, 
    fragment_video_path=fragment_video_path,
    cur_min=cur_min, 
    cur_sec=cur_sec, 
    cur_image=cur_image, 
    img_list=img_list, 
    middle_video=middle_video,
    question=question
)
answer = chat.answer(
    img_list=img_list,
    input_text=question,
    msg = msg,
    num_beams=1,
    temperature=1.0,
    max_new_tokens=300,
    max_length=2000)[0]

print(answer)
```

Note that if you receive a RuntimeError like `"Error reading <filename.mp4>"`, one solution is to initialize `<filename.mp4>` with any other video file.

## 💡 Overview

![](src/assets/overview.png)

## 📣 Demo Video

[![Alt text](https://img.youtube.com/vi/Dx5BQmgK4n8/0.jpg)](https://www.youtube.com/embed/Dx5BQmgK4n8?si=FN9pLyQBN--vJBZA)

## ⚡ Comparison Case

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Question and answer about a clip from YouTube, which is a tutorial on how to cook steak. The entire instructional process begins with marinating the steak, followed by pan-searing it, preparing side dishes, and ultimately plating the meal. Green ( Red ) highlights the correct (wrong) answer and yellow indicates that the model is hallucinating.
</div>

<p align="center" width="100%">
<a target="_blank"><img src="src/compare_case.png"  style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## 😍 Examples

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Question and answer about clips from Zootopia, a cartoon, which tells the story of a determined police officer rabbit named Judy
who pairs up with a cunning fox to uncover a conspiracy about missing animals and develop an unexpected friendship.
</div>

<p align="center" width="100%">
<a target="_blank"><img src="src/example1_00.png"  style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>


<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Question and answer about clips from Goblin, which tells the story of Kim Shin, an immortal ”goblin” who needs to find a human
bride to end his endless life but instead meets Ji Eun-tak, a girl fated to die who claims to be the ”goblin’s bride,” leading to a romantic tale unfolding bet.
</div>
<p align="center" width="100%">
<a target="_blank"><img src="src/example2_00.png" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">  Question and answer about clips from Game of Thrones, which tells the epic fantasy tale of power struggles and political intrigue among the Seven Kingdoms, entwined with intricate family relationships, all set against the backdrop of an ancient, mystical threat.
</div>
<p align="center" width="100%">
<a target="_blank"><img src="src/example3_00.png" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> Question and answer about clips from YouTube, which contains a compilation of some inspirational movies scenes. This video clip comprises several segments from The Death Crawl, Coach Carter, Rocky Balboa, and We Are Marshall,  which vary in duration.
</div>
<p align="center" width="100%">
<a target="_blank"><img src="src/example4_00.png" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## 🚀 Benchmark: MovieChat-1K 

To better evaluate the performance of MovieChat, we collect a new benchmark for long video understanding tasks, MovieChat-1K, which contains 1K high quality video clips sourced from various movies and TV series with 14K manual annotations.

To the best of our knowledge, a long video understanding dataset has not yet been established. Our work represents the initial step in creating and making it publicly available.We create MovieChat1K, containing 1k long
videos and corresponding 1k dense captions, and 13k visual question-answer pairs.For each video, we manually set and provide 1 dense caption for the whole video, 3 question-answering pairs for global mode and 10 question-answering pairs with timestamps for breakpoint mode. 

<p align="center" width="100%">
<a target="_blank"><img src="src/benchmark/dataset1.png" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>

We collect videos from 15 popular categories with varying distribution, including documentary film, detective film, animation film, and so on. Among these, each video comprises multiple alternating scenes, contributing to a diverse and dynamic visual narrative within the context of the collection. Over 90% of the videos exhibit a duration ranging from 10K to 12K frames, while 14.6% of videos extending beyond 12K frames. Only 8.6% of videos have duration less than 10k frames.


### Question-answering Pairs

#### Word Distribution
Note that MovieChat-1K is specifically designed for long video comprehension tasks, the majority of questions are open-ended, with only a quarter classified as multiple-choice questions, marked by initiators such as ‘Do,’ ‘Does,’ ‘Is,’ or ‘Are.’ We also compute the word distributions of our provided
question-answer pairs, which includes common objects (people, clothes, etc.), time (day, night, etc.), scenes (indoor, outdoor, etc.), and so on.

<p align="center" width="100%">
<a target="_blank"><img src="src/benchmark/wordcloud.png" style="width: 40%; min-width: 200px; display: block; margin: auto;"></a>

#### Sentence length distribution
MovieChat1K exhibits diverse lengths of question-answer pairs in the segmented clip level. Despite the distribution of questionanswer pairs varies between the global mode and breakpoint mode, the majority of questions tends to concentrate between 5-15 words in length, while the length of answers generally have fewer than 10 words.

<p align="center" width="100%">
<a target="_blank"><img src="src/benchmark/length.png" style="width: 70%; min-width: 200px; display: block; margin: auto;"></a>

### Dense Captions

To facilitate a more detailed understanding of long videos, we provide
a dense caption for each video. MovieChat-1K exhibits diverse caption lengths in the segmented clip level. Approximately two-thirds of the clips
have captions with 100-149 words, while one-fifth of the
clip captions have fewer than 100 words. About 11% of
clips have long captions with more than 150 words.

<p align="center" width="100%">
<a target="_blank"><img src="src/benchmark/caption_dis.png" style="width: 40%; min-width: 200px; display: block; margin: auto;"></a>

To analyze the word distribution of our generated captions, we compute their distributions. The resulting word
distribution of the captions is presented in Fig. B6, which
includes common objects (man, woman, people, girl, etc.),
attributes (detective, various, small, white, etc.), locations
(inside, behind, south, next, etc.), scenes (room, house,
building, office, etc.), actions/events (talk, enter, leave, take,
etc.), and more.

<p align="center" width="100%">
<a target="_blank"><img src="src/benchmark/caption_wordcloud.png" style="width: 45%; min-width: 200px; display: block; margin: auto;"></a>

In terms of actionness, MovieChat-1K captions contains nearly the same number of verbs as with the WebVid10M dataset. To evaluate this, we use the NLTK toolkit to
analyze the number of verbs in captions, focusing on extracting and tagging all unique verbs. We find a total of
109,485 verbs in the WebVid10M caption dataset, while the
MovieChat-1K captions contain 102,988 unique instances
of verbs. While these counts may not be entirely accurate
due to our simple counting method, we believe they provide
a rough indication of the actionness of the two datasets.

<!-- ## Comparison between MovieChat-1K and other benchmarks

MovieChat-1K provides a large-scale benchmark
for long video understanding, which contains 1K movies,
1K dense captions and 13k question-answer pairs. The
comparison between different datasets are shown in Tab. 8.
It is evident that MovieChat-1K provides the longest
average duration for movie clips. MovieQA exclusively offers question-answer pairs related to movies,
while MovieGraphs supplies captions associated with
movies. Unlike other datasets, MovieNet encompasses
three main types of texts: subtitle, synopsis, and script,
excluding question-answer pairs. Additionally, the synopsis category is designed for the entire movie rather than
video clips. Consequently, MovieChat-1K is more suitable
for studying long video comprehension compared to other
datasets.

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Dataset</th><th>Avg. Duration (min)</th><th>Number of Captions</th><th>Avg. Caption Length</th><th>Number of Question-Answer Pairs</th><th>Avg. Question Length</th><th>Avg. Answer Length</th>
    </tr>
    <tr align="center">
        <td><a href="https://arxiv.org/abs/1512.02902">MovieQA</a></td><td>3.5</td><td>-</td><td>-</td><td>14.9K</td><td>9.3</td><td>5.1</td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://arxiv.org/abs/1712.06761">MovieGraphs</a></td><td>0.73</td><td>15K</td><td>35</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr align="center">
        <td><a href="https://arxiv.org/abs/2007.10937">MovieNet</a></td><td>2.1</td><td>2.5K</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr align="center">
        <td>MovieChat-1K</td><td>9.4</td><td>1K</td><td>121</td><td>13K</td><td>7.8</td><td>2.3</td>
    </tr>
</table>
</div> -->

🔐 &#x00A9; **Due to the copyright concers and the size limitations of the movies, we  plan to release the features of the dataset. Please wait for a few weeks.**

## 🛠️ Install 

### Environment Preparation

First, create a conda environment:
```
conda env create -f environment.yml
conda activate moviechat
```

### Prerequisites

Before using the repository, make sure you have obtained the following checkpoints:

#### Pre-trained Language Decoder

- Get the original LLaMA weights in the Hugging Face format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
- Download Vicuna delta weights :point_right: [[7B](https://huggingface.co/lmsys/vicuna-7b-delta-v0)] (Note: we use **v0 weights** instead of v1.1 weights). 
- Use the following command to add delta weights to the original LLaMA weights to obtain the Vicuna weights:

```
python apply_delta.py \
    --base ckpt/LLaMA/7B_hf \
    --target ckpt/Vicuna/7B \
    --delta ckpt/Vicuna/vicuna-7b-delta-v0 \
```

#### Pre-trained Visual Encoder for MovieChat
- Download the MiniGPT-4 model (trained linear layer) from this [link](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view).

#### Download Pretrained Weights

- Download pretrained weights to run MovieChat with Vicuna-7B as language decoder locally from this [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-vicuna7b-v2.pth).

## 🤖 How to Run Demo Locally

Firstly, set the `llama_model`, `llama_proj_model` and `ckpt` in [eval_configs/MovieChat.yaml](./eval_configs/MovieChat.yaml).
Then run the script:
```
python inference.py \
    --cfg-path eval_configs/MovieChat.yaml \
    --gpu-id 0 \
    --num-beams 1 \
    --temperature 1.0 \
    --text-query "What is he doing?" \
    --video-path src/examples/Cooking_cake.mp4 \
    --fragment-video-path src/video_fragment/output.mp4 \
    --cur-min 1 \
    --cur-sec 1 \
    --middle-video 1 \
```
Note that, if you want to use the global mode (understanding and question-answering for the **whole** video), remember to change middle-video into 0.

<!-- ## 👍 Main Results
### Short video question-answering
We use several widely
used open-ended datasets: MSVD-QA, MSRVTT-QA, and ActivityNet-QA for short video question-answering tasks. The evaluation process is under the assistance of LLM with the default hyper-parameter settings. The accuracy and relative scores on a scale of 0 to 5 are reported. Compared to previous methods, MovieChat achieves comparable performance even it is not
specifically designed for short video question-answering tasks,

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Methods</th><th>LLM</th><th>Conversation</th><th>Detail Description</th><th>Complex Reasoning</th><th>All</th>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi">Chat-UniVi-7B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">Vicuna-7B</a></td><td><b>84.1</b></td><td>74.2</td><td>93.7</td><td>84.2</td>
    </tr>
    </tr>
    <tr align="center">
        <td><a href="https://huggingface.co/Chat-UniVi/Chat-UniVi-13B">Chat-UniVi-13B</a></td><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.5">Vicuna-13B</a></td><td><b>84.1</b></td><td><b>79.4</b></td><td><b>94.7</b></td><td><b>86.1</b></td>
    </tr>
</table>
</div> -->

## 🤝 Acknowledgement
We are grateful for the following awesome projects our MovieChat arising from:
* [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA): An Instruction-tuned Audio-Visual Language Model for Video Understanding
* [Token Merging](https://github.com/facebookresearch/ToMe): Your ViT but Faster
* [XMem](https://github.com/hkchengrex/XMem): Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-language Understanding with Advanced Large Language Models
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models 
* [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP): Improved Training Techniques for CLIP at Scale
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [VideoChat](https://github.com/OpenGVLab/Ask-Anything): Chat-Centric Video Understanding
* [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant


## 🔒 Term of Use
Our MovieChat is just a research preview intended for non-commercial use only. You must **NOT** use our MovieChat for any illegal, harmful, violent, racist, or sexual purposes. You are strictly prohibited from engaging in any activity that will potentially violate these guidelines. 

## ✏️ Citation

If you find MovieChat useful for your your research and applications, please cite using this BibTeX:

```bibtex
@article{song2023moviechat,
  title={MovieChat: From Dense Token to Sparse Memory for Long Video Understanding},
  author={Song, Enxin and Chai, Wenhao and Wang, Guanhong and Zhang, Yucheng and Zhou, Haoyang and Wu, Feiyang and Guo, Xun and Ye, Tian and Lu, Yan and Hwang, Jenq-Neng and others},
  journal={arXiv preprint arXiv:2307.16449},
  year={2023}
}

@article{song2024moviechat+,
  title={MovieChat+: Question-aware Sparse Memory for Long Video Question Answering},
  author={Song, Enxin and Chai, Wenhao and Ye, Tian and Hwang, Jenq-Neng and Li, Xi and Wang, Gaoang},
  journal={arXiv preprint arXiv:2404.17176},
  year={2024}
}
```
