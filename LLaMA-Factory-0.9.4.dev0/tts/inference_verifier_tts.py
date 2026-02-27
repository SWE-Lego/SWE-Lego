#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def get_yes_no_token_ids(tokenizer):
    yes_candidates = ["YES", "Yes", "yes", " YES", " Yes", " yes"]
    no_candidates = ["NO", "No", "no", " NO", " No", " no"]

    yes_ids = sorted({tokenizer.encode(x, add_special_tokens=False)[0] for x in yes_candidates if tokenizer.encode(x, add_special_tokens=False)})
    no_ids = sorted({tokenizer.encode(x, add_special_tokens=False)[0] for x in no_candidates if tokenizer.encode(x, add_special_tokens=False)})

    if not yes_ids or not no_ids:
        raise ValueError(f"Failed to resolve YES/NO token ids: yes_ids={yes_ids}, no_ids={no_ids}")

    return yes_ids, no_ids


def build_prompt(tokenizer, sample):
    messages = sample["messages"]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "<judgement>"


def score_batch(model, tokenizer, prompts, yes_ids, no_ids, max_length):
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[:, -1, :]

    yes_logits = torch.stack([logits[:, i] for i in yes_ids], dim=-1).max(dim=-1).values
    no_logits = torch.stack([logits[:, i] for i in no_ids], dim=-1).max(dim=-1).values

    # exp(yes) / (exp(yes) + exp(no))
    prob = torch.sigmoid(yes_logits - no_logits)
    return prob.detach().cpu().float().tolist()


def main():
    parser = argparse.ArgumentParser(description="Minimal verifier inference for TTS trajectories")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    data = load_data(args.input_path)

    grouped = defaultdict(list)
    for i in tqdm(range(0, len(data), args.batch_size), desc="inference"):
        batch = data[i : i + args.batch_size]
        prompts = [build_prompt(tokenizer, x) for x in batch]
        scores = score_batch(model, tokenizer, prompts, yes_ids, no_ids, args.max_length)

        for sample, pred_score in zip(batch, scores):
            instance_id = sample.get("instance_id", "unknown")
            run_id = sample.get("run_id", sample.get("run", "unknown"))
            gt_score = sample.get("ground_truth", sample.get("score", None))
            grouped[instance_id].append(
                {
                    "run": run_id,
                    "score": gt_score,
                    "predicted_score": pred_score,
                }
            )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for instance_id in sorted(grouped.keys()):
            grouped[instance_id].sort(key=lambda x: str(x["run"]))
            f.write(json.dumps({"instance_id": instance_id, "runs": grouped[instance_id]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
