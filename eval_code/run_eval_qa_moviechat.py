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

    pred_file = open(args.pred_path)
    pred_contents = json.load(pred_file)

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    openai.api_base = "https://api.aiproxy.io/v1"

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    pred_accuracy = []
    is_pred_accuracy = []
    pred_score = []
    is_pred_score = []
    for sample in pred_contents:
        id = sample.split(".")[0]
        qa_list = pred_contents[sample]
        for qa_pair in qa_list:
            question = qa_pair['question']
            answer = qa_pair['answer']
            pred = qa_pair['pred'].replace("</s>", "")
            qa_set = {"q": question, "a": answer, "pred": pred}
            prediction_set[id] = qa_set
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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
                if check_sentence_start(question):
                    if 'Yes' in pred or 'No' in pred:
                        if remove_punctuation(answer) in pred:
                            accuracy_string = 'yes'
                        else:
                            accuracy_string = 'no'
                    start_index = response_message.find("'score': ")
                    end_index = response_message.find("}", start_index)
                    if start_index != -1 and end_index != -1:
                        score_string = response_message[start_index + len("'score': "):end_index]
                    if (accuracy_string == 'yes' and float(score_string) >=2.9) or (accuracy_string == 'no' and float(score_string) <= 2.9):
                        is_pred_accuracy.append(accuracy_string)
                        print(accuracy_string)
                        is_pred_score.append(score_string)
                        print(score_string)
                        with open('mv_is_example.txt', 'a') as file:
                            file.write(accuracy_string)
                            file.write('\n')
                            file.write(score_string)
                            file.write('\n')
                else:
                    if remove_punctuation(answer) in pred or remove_punctuation(answer.lower()) in pred:
                            accuracy_string = 'yes'
                    start_index = response_message.find("'score': ")
                    end_index = response_message.find("}", start_index)
                    if start_index != -1 and end_index != -1:
                        score_string = response_message[start_index + len("'score': "):end_index]
                    if (accuracy_string == 'yes' and float(score_string) >=2.9) or (accuracy_string == 'no' and float(score_string) <= 2.9):
                        pred_accuracy.append(accuracy_string)
                        print(accuracy_string)
                        pred_score.append(score_string)
                        print(score_string)
                        with open('mv_example.txt', 'a') as file:
                            file.write(accuracy_string)
                            file.write('\n')
                            file.write(score_string)
                            file.write('\n')
                # with open('mv_result.txt', 'a') as file:
                #     file.write(accuracy_string)
                #     file.write('\n')
                #     file.write(score_string)
                #     file.write('\n')



      


if __name__ == "__main__":
    main()

    