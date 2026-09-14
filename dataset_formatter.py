"""Dataset loading and tokenisation. TREC comes from the Hub, the rest from `data_root`."""

import os

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

JSON_MAX_LENGTH = 256
TREC_MAX_LENGTH = 128


def clean_instruction(text: str) -> str:
    """Normalise the answer marker and drop everything after it."""
    text = text.replace("\n\nAnswer format:", "\n\nAnswer:")
    text = text.replace("\n\nAnswer Format:", "\n\nAnswer:")

    if "\n\nAnswer:" in text:
        text = text.split("\n\nAnswer:")[0].strip() + "\n\nAnswer:"
    return text


def clean_instruction_arc_trec(text: str) -> str:
    """Reduce an ARC/OpenBookQA instruction to the bare question."""
    if "\n\nAnswer1:" in text:
        text = text.split("\n\nAnswer1:")[0].strip()
    if "to the question: " in text:
        text = text.split("to the question: ")[-1].strip()
    return text


def _finalize(train_dataset, valid_dataset):
    columns = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    return train_dataset, valid_dataset


def _load_trec(tokenizer: PreTrainedTokenizerBase):
    raw = load_dataset("trec", trust_remote_code=True)

    label_feature = raw["train"].features["coarse_label"]
    id2label = {i: name for i, name in enumerate(label_feature.names)}
    label2id = {name: i for i, name in id2label.items()}

    def tokenize(examples):
        texts = [f"Question: {t}\nLabel:" for t in examples["text"]]
        model_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=TREC_MAX_LENGTH,
            return_tensors="pt",
        )
        model_inputs["label"] = examples["coarse_label"]
        return model_inputs

    train_dataset, valid_dataset = _finalize(
        raw["train"].map(tokenize, batched=True),
        raw["test"].map(tokenize, batched=True),
    )
    return train_dataset, valid_dataset, len(id2label), label2id, id2label


def prepare_datasets(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    data_root: str = "dataset_all",
):
    """Return train_dataset, valid_dataset, num_labels, label2id, id2label."""
    if dataset_name == "trec":
        return _load_trec(tokenizer)

    data_files = {
        "train": os.path.join(data_root, dataset_name, "train.json"),
        "validation": os.path.join(data_root, dataset_name, "test.json"),
    }
    raw = load_dataset("json", data_files=data_files)

    all_answers = sorted(set(raw["train"]["answer"]) | set(raw["validation"]["answer"]))
    label2id = {ans: i for i, ans in enumerate(all_answers)}
    id2label = {i: ans for ans, i in label2id.items()}

    def tokenize(examples):
        cleaned = [clean_instruction(instr) for instr in examples["instruction"]]
        model_inputs = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=JSON_MAX_LENGTH,
            return_tensors="pt",
        )
        model_inputs["label"] = [label2id[a] for a in examples["answer"]]
        return model_inputs

    train_dataset, valid_dataset = _finalize(
        raw["train"].map(tokenize, batched=True),
        raw["validation"].map(tokenize, batched=True),
    )
    return train_dataset, valid_dataset, len(label2id), label2id, id2label


def prepare_datasets_calib(
    dataset_name: str,
    eval_dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    data_root: str = "dataset_all",
):
    """Load `dataset_name` using `eval_dataset_name`'s prompt template."""
    if dataset_name == "trec":
        return _load_trec(tokenizer)

    data_files = {
        "train": os.path.join(data_root, dataset_name, "train.json"),
        "validation": os.path.join(data_root, dataset_name, "test.json"),
    }
    raw = load_dataset("json", data_files=data_files)

    all_answers = sorted(set(raw["train"]["answer"]) | set(raw["validation"]["answer"]))
    label2id = {ans: i for i, ans in enumerate(all_answers)}
    id2label = {i: ans for ans, i in label2id.items()}

    to_trec_format = eval_dataset_name == "trec" and dataset_name in (
        "arc_easy",
        "arc_challenge",
        "openbookqa",
    )

    def tokenize(examples):
        cleaned = [clean_instruction(instr) for instr in examples["instruction"]]
        if to_trec_format:
            cleaned = [f"Question: {clean_instruction_arc_trec(c)}\nLabel:" for c in cleaned]

        model_inputs = tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=JSON_MAX_LENGTH,
            return_tensors="pt",
        )
        model_inputs["label"] = [label2id[a] for a in examples["answer"]]
        return model_inputs

    train_dataset, valid_dataset = _finalize(
        raw["train"].map(tokenize, batched=True, load_from_cache_file=False),
        raw["validation"].map(tokenize, batched=True, load_from_cache_file=False),
    )
    return train_dataset, valid_dataset, len(label2id), label2id, id2label
