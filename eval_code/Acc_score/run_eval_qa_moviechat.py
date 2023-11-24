"""
Adapted from: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/evaluate_activitynet_qa.py
"""
import openai
import os
import re
import argparse
import json
import ast
import string


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    return args

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    output_dir = args.output_dir

    pred_file = open(args.pred_path)
    pred_contents = json.load(pred_file)

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    openai.api_base = "https://api.aiproxy.io/v1"

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    count = 0
    for sample in pred_contents:
        id = sample.split(".")[0]
        qa_list = pred_contents[sample]
        for qa_pair in qa_list:
            question = qa_pair['question']
            answer = qa_pair['answer']
            pred = qa_pair['pred'].replace("</s>", "")
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set
            count += 1
            if count > 0:
                completion = openai.ChatCompletion.create(
                    model="claude-instant-1",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                
                start_index = response_message.find("'pred': '")
                end_index = response_message.find("',", start_index)
                if start_index != -1 and end_index != -1:
                    accuracy_string = response_message[start_index + len("'pred': '"):end_index]
                    print(accuracy_string)
                    
                start_index = response_message.find("'score': ")
                end_index = response_message.find("}", start_index)
                if start_index != -1 and end_index != -1:
                    score_string = response_message[start_index + len("'score': "):end_index]
                    print(score_string)

      


if __name__ == "__main__":
    main()

    
