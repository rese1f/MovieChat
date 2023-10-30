"""
Adapted from: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/evaluate_activitynet_qa.py
"""
import openai
import os
import re
import argparse
import json
import ast


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
    count = 0

    with open(output_dir, 'a') as output_json_file:

        # Set the OpenAI API key.
        openai.api_key = args.api_key
        openai.api_base = "https://api.aiproxy.io/v1"

        # Preparing dictionary of question-answer sets
        for sample in pred_contents:
            count += 1
            if count > 18:
                # id = sample.split(".")[0]
                question = sample['question']
                answer = sample['answer']
                pred = sample['pred']
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": 
                                "You function as an intelligent chatbot, with the purpose of summarizing the content of the predicted answer."
                                "Your goal is to shorten the length of the predicted answer to fewer than 10 words while preserving its original meaning, based on the question. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Simplify the Predicted Answer into a process answer of no more than 10 words."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please simplify the Predicted Answer:\n\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your summarized outcome, ensuring it remains within 10 words."
                                "Please generate the response in the form of a Python dictionary string with keys 'summarized_answer', where value of 'pred' is  a string."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'summarized_answer': 'Yes.'}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion["choices"][0]["message"]["content"]
                start_index = response_message.find("'summarized_answer': '")
                end_index = response_message.find("'}", start_index)
                if start_index != -1 and end_index != -1:
                    summarized_answer = response_message[start_index + len("'summarized_answer': '"):end_index]
                    print(summarized_answer)
                    result_data = {}
                    result_data['question'] = question
                    result_data['answer'] = answer
                    result_data['pred'] = summarized_answer
                    output_json_file.write(json.dumps(result_data))
                    output_json_file.write("\n")
            
            
if __name__ == "__main__":
    main()

    