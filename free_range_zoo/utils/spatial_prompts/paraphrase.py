import sys
import argparse
from pathlib import Path
import openai
# import anthropic

openai.api_key = "**********************"
openai.api_base = "https://api.lqqq.ltd/v1"


def paraphrase(string, num=1):
    messages = [
        {"role": "system", "content": "Please paraphrase the following instructions while preserving their meaning. Any words surrounded by {} should also appear in your result with a similar context."},
        {"role": "user", "content": string}
    ]
    print('before response')
    responses = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-16k-0613",
        model="gpt-4-0613",
        messages=messages,
        temperature=0.7,
        n=num,
    )
    print('after response')

    return [choice["message"]["content"] for choice in responses["choices"]]

if __name__ == "__main__":
    """
    Example usage:
    python paraphrase.py initial_system.txt -n 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to file containing content to paraphrase")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of paraphrases to generate")
    args = parser.parse_args()
    filename, num = Path(args.filename), args.num

    with open(filename, "r") as f:
        responses = paraphrase(f.read(), num)
    for i, response in enumerate(responses):
        with open(filename.parent / Path(str(filename.stem) + f"-{i}" + str(filename.suffix)), "w") as f:
            f.write(response)
