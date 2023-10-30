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

def check_sentence_start(sentence):
    sentence_lower = sentence.lower()

    if sentence_lower.startswith("does") or sentence_lower.startswith("do") or sentence_lower.startswith("is") or sentence_lower.startswith("are"):
        return True
    else:
        return False

def remove_punctuation(sentence):
    punctuation_set = set(string.punctuation)
    sentence_no_punct = ''.join(char for char in sentence if char not in punctuation_set)

    return sentence_no_punct

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    output_dir = args.output_dir

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    openai.api_base = "https://api.aiproxy.io/v1"

    # Preparing dictionary of question-answer sets

    with open(args.pred_path, 'r') as txt_file:
        for line in txt_file:
            pred_contents = json.loads(line.strip())
            question = pred_contents['question']
            answer = pred_contents['answer']
            pred = pred_contents['pred']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                            "- Evaluate the temporal accuracy of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            # response_dict = ast.literal_eval(response_message)
            with open('temporal.txt', 'a') as file:
                file.write(response_message)
                file.write('\n')





if __name__ == "__main__":
    main()

    