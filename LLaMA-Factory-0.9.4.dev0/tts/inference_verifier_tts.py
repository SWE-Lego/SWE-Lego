#!/usr/bin/env python3
"""Run generative verifier inference for TTS trajectories.

The script scores each trajectory with:
  P(YES) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
and outputs aggregated JSONL grouped by ``instance_id``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def apply_liger_kernel_for_inference(config: AutoConfig, enable: bool = False) -> None:
    if not enable:
        return

    model_type = getattr(config, "model_type", None)
    try:
        if model_type == "qwen2":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2()
        elif model_type == "qwen3":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3

            apply_liger_kernel_to_qwen3()
        elif model_type == "qwen3_moe":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

            apply_liger_kernel_to_qwen3_moe()
        elif model_type == "llama":
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            apply_liger_kernel_to_llama()
        elif model_type == "mistral":
            from liger_kernel.transformers import apply_liger_kernel_to_mistral

            apply_liger_kernel_to_mistral()
        elif model_type == "gemma":
            from liger_kernel.transformers import apply_liger_kernel_to_gemma

            apply_liger_kernel_to_gemma()
        elif model_type == "gemma2":
            from liger_kernel.transformers import apply_liger_kernel_to_gemma2

            apply_liger_kernel_to_gemma2()
        else:
            print(f"[warning] Liger kernel does not support model_type={model_type}, skipping.")
            return
        print(f"[info] Liger kernel enabled for model_type={model_type}.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[warning] Failed to enable liger kernel: {exc}")


def get_attn_implementation(flash_attn: str, rank: int = 0) -> Optional[str]:
    if flash_attn == "auto":
        return None
    if flash_attn == "disabled":
        if rank == 0:
            print("[info] Using eager attention")
        return "eager"
    if flash_attn == "sdpa":
        if not is_torch_sdpa_available():
            if rank == 0:
                print("[warning] SDPA not available, fallback to auto")
            return None
        if rank == 0:
            print("[info] Using SDPA")
        return "sdpa"
    if flash_attn == "fa2":
        if not is_flash_attn_2_available():
            if rank == 0:
                print("[warning] FlashAttention-2 not available, fallback to auto")
            return None
        if rank == 0:
            print("[info] Using FlashAttention-2")
        return "flash_attention_2"

    if rank == 0:
        print(f"[warning] Unknown flash_attn={flash_attn}, fallback to auto")
    return None


class TTSVerifierDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 131072):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        messages = sample["messages"]
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # pylint: disable=broad-except
                text = self._manual_format(messages)
        else:
            text = self._manual_format(messages)

        return {
            "text": text,
            "original_data": sample,
        }

    @staticmethod
    def _manual_format(messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts) + "\n<|im_start|>assistant\n"


def get_yes_no_token_ids(tokenizer: AutoTokenizer) -> Tuple[List[int], List[int]]:
    yes_candidates = ["YES", "Yes", "yes", " YES", " Yes", " yes"]
    no_candidates = ["NO", "No", "no", " NO", " No", " no"]

    yes_ids = []
    no_ids = []

    for token in yes_candidates:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            yes_ids.append(ids[0])

    for token in no_candidates:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            no_ids.append(ids[0])

    yes_ids = sorted(set(yes_ids))
    no_ids = sorted(set(no_ids))
    if not yes_ids or not no_ids:
        raise ValueError(f"Cannot find YES/NO token IDs. yes_ids={yes_ids}, no_ids={no_ids}")

    return yes_ids, no_ids


def compute_yes_no_probability(
    logits: torch.Tensor,
    yes_ids: List[int],
    no_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yes_logits = torch.stack([logits[:, token_id] for token_id in yes_ids], dim=-1)
    no_logits = torch.stack([logits[:, token_id] for token_id in no_ids], dim=-1)

    max_yes_logit, _ = yes_logits.max(dim=-1)
    max_no_logit, _ = no_logits.max(dim=-1)

    max_logit = torch.maximum(max_yes_logit, max_no_logit)
    exp_yes = torch.exp(max_yes_logit - max_logit)
    exp_no = torch.exp(max_no_logit - max_logit)

    normalized_yes_prob = exp_yes / (exp_yes + exp_no)
    return normalized_yes_prob, max_yes_logit, max_no_logit


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    judgement_prefix: str,
) -> Dict[str, Any]:
    texts = [item["text"] + judgement_prefix for item in batch]
    tokenizer.padding_side = "left"
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "original_data": [item["original_data"] for item in batch],
    }


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _run_sort_key(run_value: Any) -> Tuple[int, str]:
    try:
        return (0, str(int(run_value)))
    except Exception:  # pylint: disable=broad-except
        return (1, str(run_value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative verifier inference for TTS trajectories")
    parser.add_argument("--model_path", type=str, required=True, help="Verifier checkpoint path or HF model id")
    parser.add_argument("--data_path", type=str, required=True, help="Input trajectories (.json or .jsonl)")
    parser.add_argument("--output_path", type=str, required=True, help="Output aggregated predictions (.jsonl)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--max_length", type=int, default=131072, help="Maximum input length")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 for model weights")
    parser.add_argument(
        "--flash_attn",
        type=str,
        default="fa2",
        choices=["auto", "disabled", "sdpa", "fa2"],
        help="Attention backend",
    )
    parser.add_argument("--enable_liger_kernel", action="store_true", help="Enable liger-kernel optimizations")
    parser.add_argument("--clear_cache_steps", type=int, default=10, help="Clear GPU cache every N steps")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"[info] world_size={world_size}")
        print(f"[info] model={args.model_path}")
        print(f"[info] data={args.data_path}")
        print(f"[info] output={args.output_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    yes_ids, no_ids = get_yes_no_token_ids(tokenizer)
    if rank == 0:
        print(f"[info] yes_ids={yes_ids}, no_ids={no_ids}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling is None:
        config.rope_scaling = {"type": "yarn", "factor": 4.0}

    apply_liger_kernel_for_inference(config, enable=args.enable_liger_kernel)
    attn_implementation = get_attn_implementation(args.flash_attn, rank)

    model_kwargs: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()

    dataset = TTSVerifierDataset(args.data_path, tokenizer, args.max_length)

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, "<judgement>"),
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length, "<judgement>"),
            num_workers=0,
        )

    iterator = tqdm(dataloader, desc=f"rank{rank}") if rank == 0 else dataloader

    all_results: List[Dict[str, Any]] = []
    step_count = 0

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :]
            pred_scores, logit_yes, logit_no = compute_yes_no_probability(logits, yes_ids, no_ids)

            pred_scores = pred_scores.detach().cpu().float().tolist()
            logit_yes = logit_yes.detach().cpu().float().tolist()
            logit_no = logit_no.detach().cpu().float().tolist()

            for i, original_data in enumerate(batch["original_data"]):
                run_id = original_data.get("run_id", original_data.get("run", "unknown"))
                score = original_data.get("ground_truth", original_data.get("score", None))
                all_results.append(
                    {
                        "instance_id": original_data.get("instance_id", f"sample_{i}"),
                        "run": run_id,
                        "score": score,
                        "predicted_score": pred_scores[i],
                        "logit_yes": logit_yes[i],
                        "logit_no": logit_no[i],
                    }
                )

            del outputs, logits, input_ids, attention_mask
            step_count += 1
            if args.clear_cache_steps > 0 and step_count % args.clear_cache_steps == 0:
                clear_gpu_memory()

    if world_size > 1:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, all_results)
        if rank == 0:
            all_results = []
            for part in gathered_results:
                all_results.extend(part)

    if rank == 0:
        grouped = defaultdict(list)
        for item in all_results:
            grouped[item["instance_id"]].append(
                {
                    "run": item["run"],
                    "score": item["score"],
                    "predicted_score": item["predicted_score"],
                    "logit_yes": item["logit_yes"],
                    "logit_no": item["logit_no"],
                }
            )

        for instance_id in grouped:
            grouped[instance_id].sort(key=lambda x: _run_sort_key(x["run"]))

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for instance_id in sorted(grouped.keys()):
                row = {"instance_id": instance_id, "runs": grouped[instance_id]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[done] wrote {len(grouped)} instances ({len(all_results)} runs) to {output_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
