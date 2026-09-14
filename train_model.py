"""LoRA fine-tuning of the CacheTrap victim classifiers."""

import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset_formatter import prepare_datasets

MODEL_REGISTRY = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral7B": "mistralai/Mistral-7B-v0.1",
    "qwen3B": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek_qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

DATASETS = ["trec", "arc_easy", "openbookqa", "arc_challenge"]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a victim classifier with LoRA.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", default="dataset_all")
    parser.add_argument("--output_dir", default="./TrainedModels")
    parser.add_argument("--save_dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument("--save_lora", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_epochs(model_short: str, dataset: str) -> int:
    if dataset == "arc_challenge":
        return 6
    if dataset == "arc_easy" and model_short == "llama2":
        return 6
    return 3


def main():
    args = parse_args()
    set_global_determinism(args.seed)

    model_name = MODEL_REGISTRY[args.model]
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset, valid_dataset, num_labels, label2id, id2label = prepare_datasets(
        args.dataset, tokenizer, data_root=args.data_root
    )
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=24)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"num_labels: {num_labels}")
    print(f"label2id: {label2id}")
    print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    epochs = args.epochs if args.epochs is not None else default_epochs(args.model, args.dataset)
    print(f"Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"],
                )
                loss = outputs.loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        print(f"Epoch {epoch + 1} finished. Average Loss: {total_loss / len(train_dataloader):.4f}")

        model.eval()
        correct = 0
        total_seen = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc="Validating"):
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == batch["label"]).sum().item()
                total_seen += batch["label"].size(0)

        print(f"Validation Accuracy: {correct / total_seen if total_seen else 0.0:.4f}")

    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(model)

    if args.save_lora:
        lora_dir = os.path.join(args.output_dir, f"lora_{args.model}_{args.dataset}")
        unwrapped.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        print(f"LoRA adapter saved to {lora_dir}")

    merged_model = unwrapped.merge_and_unload().to(DTYPES[args.save_dtype])

    merged_dir = os.path.join(args.output_dir, f"merged_{args.model}_{args.dataset}")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged {args.save_dtype} checkpoint saved to {merged_dir}")


if __name__ == "__main__":
    main()
