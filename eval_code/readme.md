Follow these steps to conduct the evaluation:

**Step 1:** Run the inference. You'll need the following:
Run the command:

For MSVD-QA dataset.
```shell
python run_inference_qa_msvd.py \
    --cfg-path eval_configs/MovieChat.yaml \
    --gpu-id 0 \
    --num-beams 1 \
    --temperature 1.0 \
    --video-path /path/to/your/video \
    --gt_file /path/to/your/question and answer file \
    --output_dir /path/to/your/output \
    --output_name msvd-qa \
    --fragment-video-path src/video_fragment/output.mp4 \
```

For MSRVTT-QA dataset.
```shell
python run_inference_qa_msrvtt.py \
    --cfg-path eval_configs/MovieChat.yaml \
    --gpu-id 0 \
    --num-beams 1 \
    --temperature 1.0 \
    --video-path /path/to/your/video \
    --gt_file /path/to/your/question and answer file \
    --output_dir /path/to/your/output \
    --output_name msrvtt-qa \
    --fragment-video-path src/video_fragment/output.mp4 \
```

For ActivityNet-QA dataset.
```shell
python run_inference_qa_activitynet.py \
    --cfg-path eval_configs/MovieChat.yaml \
    --gpu-id 0 \
    --num-beams 1 \
    --temperature 1.0 \
    --video-path /path/to/your/video \
    --gt_file /path/to/your/question file \
    --gt_file_answers /path/to/your/answer file \
    --output_dir /path/to/your/output \
    --output_name activitynet-qa \
    --fragment-video-path src/video_fragment/output_act.mp4 \
```
This will generate a JSON file containing the model's predicted responses.

**Step 2:** Evaluate the predicted responses. The evaluation process computes the accuracy and assigns a score on a scale of 1-5. This step requires the predictions from step-1, question-answer pair annotations, and an OpenAI API key.

Run the command:

```shell
python run_eval_qa.py \
    --pred_path <qa_preds> \
    --output_dir <path-to-out-dir> \
    --output_json <qa_results> \
    --api_key <your-openai-api_key> \
    --num_tasks 1
```