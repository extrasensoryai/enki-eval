import re
from tqdm import tqdm
import json
import argparse
import datasets
import requests
from openai import OpenAI
import anthropic
from llama_cpp import Llama
from settings import OPENAI_KEY, ANTHROPIC_KEY, FIREWORKS_KEY
from typing import cast, List, Dict

# Enuemrate the question, response columns

prompt = """
Answer this question return only a numbered list of answers. Do not include
any other text in your response.

{question}
"""

abstention_words = [
    "sorry" "apologize",
    "cannot" "unable",
]

SYSTEM_PROMPT = """
The following is a biochemistry question. Let's think step by step.
""".strip()


def detect_abstenstion(output):
    for word in abstention_words:
        if word in output:
            return True
    return False


def convert_list(output):
    if detect_abstenstion(output):
        return []
    else:
        final = [x[2:].strip() for x in output.strip().split("\n")]
    return [x for x in final if len(x.strip()) > 0]


def get_openai_response(question):
    fprompt = prompt.format(question=question)
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": fprompt}]
    )
    return response.choices[0].message.content


def get_anthropic_response(question):
    fprompt = prompt.format(question=question)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": fprompt}],
        max_tokens=256,
    )
    return response.content[0].text  # type: ignore


def create_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def get_model_response(llm: Llama, messages: List[Dict[str, str]]) -> str:
    temperature = 0.7
    top_p = 0.9
    response = llm.create_chat_completion(
        messages=messages,  # type: ignore
        temperature=temperature,
        top_p=top_p,
        max_tokens=2048,
    )
    return response["choices"][0]["message"]["content"]  # type: ignore


def get_llama_response(question: str) -> str:
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    fprompt = prompt.format(question=question)

    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "max_tokens": 16384,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {"role": "user", "content": fprompt},
        ],
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_KEY}",
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    completion = cast(str, response.json()["choices"][0]["message"]["content"])
    return completion


def compute_hits_at_k(predictions, golden_answers):
    hits = 0
    for pred, gold in zip(predictions, golden_answers):
        lower_preds = [p.lower() for p in pred]
        if any(answer.lower() in lower_preds for answer in gold):
            hits += 1
    return hits / len(predictions) if predictions else 0


def compute_f1_score(predictions, golden_answers):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gold in zip(predictions, golden_answers):
        pred_set = set([p.lower() for p in pred])
        gold_set = set([g.lower() for g in gold])

        true_positives += len(pred_set.intersection(gold_set))
        false_positives += len(pred_set - gold_set)
        false_negatives += len(gold_set - pred_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


def extract_answers(response):
    # Extract all answers after the <|endthought|> tag
    match = re.search(r"<\|endthought\|>\s*(.*)", response, re.DOTALL)
    if match:
        answers = match.group(1).strip().split("\n")
        return [answer.strip() for answer in answers if answer.strip()]
    return []


def compute_accuracy(predictions, golden_answers):
    correct = 0
    total = len(predictions)
    for pred, gold in zip(predictions, golden_answers):
        pred_set = set([p.lower() for p in pred])
        gold_set = set([g.lower() for g in gold])
        if pred_set == gold_set:
            correct += 1
    return correct / total if total > 0 else 0


def driver(count, model_path):
    ds = datasets.load_dataset("extrasensory/enki")

    df = ds["test"].to_pandas()

    # slice off the count rows
    df = df[0:count]

    openai_predictions = []
    anthropic_predictions = []
    llamas_predictions = []
    model_predictions = []

    golden_answers_list = []

    llm = Llama(model_path=model_path)

    for i, row in tqdm(df.iterrows(), total=min(count, len(df))):
        question = row["question"]
        response = row["response"]
        golden_answers = extract_answers(response)

        # Benchmark models
        openai_pred = convert_list(get_openai_response(question))
        anthropic_pred = convert_list(get_anthropic_response(question))
        llamas_pred = convert_list(get_llama_response(question))

        # Our model
        model_pred = convert_list(get_model_response(llm, create_messages(question)))

        openai_predictions.append(openai_pred)
        anthropic_predictions.append(anthropic_pred)
        llamas_predictions.append(llamas_pred)
        model_predictions.append(model_pred)

        golden_answers_list.append(golden_answers)

        # tqdm.write(f"Question: {question}")
        # tqdm.write(f"Ground Truth: {golden_answers}")
        # tqdm.write(f"OpenAI Prediction: {openai_pred}")
        # tqdm.write(f"Anthropic Prediction: {anthropic_pred}")
        # tqdm.write(f"llama-70b Prediction: {llamas_pred}")
        # tqdm.write(f"Model Prediction: {model_pred}")
        # tqdm.write("---")

    openai_hits_at_k = compute_hits_at_k(openai_predictions, golden_answers_list)
    anthropic_hits_at_k = compute_hits_at_k(anthropic_predictions, golden_answers_list)
    llamas_hits_at_k = compute_hits_at_k(llamas_predictions, golden_answers_list)
    model_hits_at_k = compute_hits_at_k(model_predictions, golden_answers_list)

    openai_precision, openai_recall, openai_f1 = compute_f1_score(
        openai_predictions, golden_answers_list
    )
    anthropic_precision, anthropic_recall, anthropic_f1 = compute_f1_score(
        anthropic_predictions, golden_answers_list
    )
    llamas_precision, llamas_recall, llamas_f1 = compute_f1_score(
        llamas_predictions, golden_answers_list
    )
    model_precision, model_recall, model_f1 = compute_f1_score(
        model_predictions, golden_answers_list
    )

    openai_accuracy = compute_accuracy(openai_predictions, golden_answers_list)
    anthropic_accuracy = compute_accuracy(anthropic_predictions, golden_answers_list)
    llamas_accuracy = compute_accuracy(llamas_predictions, golden_answers_list)
    model_accuracy = compute_accuracy(model_predictions, golden_answers_list)

    tqdm.write("-" * 50)
    tqdm.write(f"Total Questions: {len(df)}")

    tqdm.write("-" * 50)
    tqdm.write(f"OpenAI Hits@k: {openai_hits_at_k:.4f}")
    tqdm.write(f"OpenAI Accuracy: {openai_accuracy:.4f}")
    tqdm.write(f"OpenAI Precision: {openai_precision:.4f}")
    tqdm.write(f"OpenAI Recall: {openai_recall:.4f}")
    tqdm.write(f"OpenAI F1: {openai_f1:.4f}")

    tqdm.write("-" * 50)
    tqdm.write(f"Anthropic Hits@k: {anthropic_hits_at_k:.4f}")
    tqdm.write(f"Anthropic Accuracy: {anthropic_accuracy:.4f}")
    tqdm.write(f"Anthropic Precision: {anthropic_precision:.4f}")
    tqdm.write(f"Anthropic Recall: {anthropic_recall:.4f}")
    tqdm.write(f"Anthropic F1: {anthropic_f1:.4f}")

    tqdm.write("-" * 50)
    tqdm.write(f"llama-70b Hits@k: {llamas_hits_at_k:.4f}")
    tqdm.write(f"llama-70b Accuracy: {llamas_accuracy:.4f}")
    tqdm.write(f"llama-70b Precision: {llamas_precision:.4f}")
    tqdm.write(f"llama-70b Recall: {llamas_recall:.4f}")
    tqdm.write(f"llama-70b F1: {llamas_f1:.4f}")

    tqdm.write("-" * 50)
    tqdm.write(f"Custom Model Hits@k: {model_hits_at_k:.4f}")
    tqdm.write(f"Custom Model Accuracy: {model_accuracy:.4f}")
    tqdm.write(f"Custom Model Precision: {model_precision:.4f}")
    tqdm.write(f"Custom Model Recall: {model_recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("model_path", type=str, default="./med_llama_8b.gguf")
    args = parser.parse_args()

    driver(args.count, args.model_path)
