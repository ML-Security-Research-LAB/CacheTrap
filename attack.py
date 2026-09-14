"""CacheTrap: single-bit flips in the KV cache of a victim classifier."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from dataset_formatter import prepare_datasets, prepare_datasets_calib

MODEL_REGISTRY = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral7B": "mistralai/Mistral-7B-v0.1",
    "qwen3B": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek_qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

DATASETS = ["trec", "arc_easy", "openbookqa", "arc_challenge"]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

FLIP_BIT_POS = 14

# ======================================
#   Bit-flip helpers
# ======================================

def flip_exponent_bit(value: float) -> float:
    """
    Flip the MSB-1 exponent bit of a float16 value.
    """
    float16_val = np.float16(value)
    int16_bits = np.frombuffer(float16_val.tobytes(), dtype=np.uint16)[0]
    bit_mask = 1 << FLIP_BIT_POS
    flipped_bits = int16_bits ^ bit_mask
    flipped_bytes = flipped_bits.tobytes()
    flipped_float16 = np.frombuffer(flipped_bytes, dtype=np.float16)[0]
    return float(flipped_float16)


def collect_kv_values(
    dataloader,
    base_model,
    device,
    max_samples: int = 50,
    layers=None,
    corrupt_k: int = 1,
):
    """
    Collect KV values for token position n-k in prefix-only runs,
    for up to `max_samples` examples, optionally restricting to certain layers.

    Returns:
        kv_values: dict[(layer, head, dim)] -> list[float]
    """
    kv_values = {}
    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                if samples_collected >= max_samples:
                    return kv_values

                ids = input_ids[i:i + 1]
                mask = attention_mask[i:i + 1]

                seq_len = mask.sum(dim=1).item()
                last_idx = seq_len - 1
                if last_idx == 0:
                    continue

                prefix_ids = ids[:, :last_idx]
                prefix_mask = mask[:, :last_idx]

                prefix_outputs = base_model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = prefix_outputs.past_key_values

                for layer_idx, (k, v) in enumerate(past_kv):
                    if layers is not None and layer_idx not in layers:
                        continue

                    # v: [batch=1, num_heads, seq_len_prefix, head_dim]
                    seq_len_prefix = v.shape[2]
                    target_pos = seq_len_prefix - corrupt_k
                    if target_pos < 0:
                        continue

                    target_token = v[0, :, target_pos, :]  # [num_heads, head_dim]
                    num_heads, head_dim = target_token.shape

                    for head in range(num_heads):
                        for dim in range(head_dim):
                            key = (layer_idx, head, dim)
                            value = target_token[head, dim].item()
                            if key not in kv_values:
                                kv_values[key] = []
                            kv_values[key].append(value)

                samples_collected += 1

    return kv_values


def calibrate_positions_for_layer(
    calib_dataloader,
    base_model,
    device,
    n_calib_samples: int = 50,
    top_k: int = 1,
    layers=None,
    top_bottom: str = "top",
    corrupt_k: int = 1,
):
    """
    For the specified layer(s), compute importance scores for KV positions
    based on L2 norm of their collected values and return the top_k positions.

    NOTE (new semantics):
    - top_k now means: we return the *top_k candidate positions* for this layer,
      but they will be tried *one at a time* (single-bit flip) later.
    """
    print("\n[Calibration] Collecting KV values for position ranking...")
    kv_values = collect_kv_values(
        calib_dataloader,
        base_model,
        device,
        max_samples=n_calib_samples,
        layers=layers,
        corrupt_k=corrupt_k,
    )

    print("[Calibration] Computing L2-based importance scores...")
    importance_scores = {}

    for key, values in kv_values.items():
        # L2 norm across samples
        l2_all = np.sqrt(np.sum(np.square(values)))
        importance_scores[key] = l2_all

    if not importance_scores:
        print("[Calibration] No KV values collected for these layers.")
        return []

    reverse = (top_bottom == "top")
    sorted_positions = sorted(importance_scores.items(), key=lambda x: x[1], reverse=reverse)
    top_positions = sorted_positions[:top_k]

    corruption_positions = []
    for (layer, head, dim), score in top_positions:
        corruption_positions.append({"layer": layer, "head": head, "dim": dim, "score": score})

    print(f"[Calibration] Selected top-{top_k} candidate positions ({top_bottom}) for these layers:")
    for i, pos in enumerate(corruption_positions, 1):
        print(
            f"  {i}. Layer {pos['layer']}, Head {pos['head']}, "
            f"Dim {pos['dim']}, Score={pos['score']:.4f}"
        )

    return corruption_positions


def corrupt_past_kv(past_kv, corruption_positions, corrupt_k: int = 1):
    """
    Apply bit flips at selected KV positions in v.

    In our new setup, we will typically pass a list with a single position,
    so there is at most *one* bit flip per forward, as requested.
    """
    new_past_kv = []
    positions_by_layer = {}

    for pos in corruption_positions:
        positions_by_layer.setdefault(pos["layer"], []).append(pos)

    for layer_idx, (k, v) in enumerate(past_kv):
        k_new = k.detach().clone()
        v_new = v.detach().clone()
        seq_len = v_new.shape[2]
        target_token_pos = seq_len - corrupt_k
        if target_token_pos < 0:
            new_past_kv.append((k_new, v_new))
            continue

        if layer_idx in positions_by_layer:
            for pos in positions_by_layer[layer_idx]:
                head = pos["head"]
                dim = pos["dim"]
                current_value = v_new[0, head, target_token_pos, dim].item()
                v_new[0, head, target_token_pos, dim] = flip_exponent_bit(current_value)

        new_past_kv.append((k_new, v_new))

    return tuple(new_past_kv)


# ======================================
#   Evaluation logic
# ======================================

def evaluate_attack_all_classes(
    dataloader,
    base_model,
    classifier,
    full_model,
    device,
    corruption_positions,
    num_labels: int,
    corrupt_k: int = 1,
    max_samples=None,
    desc: str = "Bit-Flip Attack Eval",
):
    """
    Evaluate KV attack with given corruption positions.

    Returns:
        acc: float (overall accuracy under attack)
        asr_per_label: np.ndarray[num_labels]
                       asr_per_label[c] = fraction of total samples
                                          classified as class c
    """
    print(f"\n[Bit-Flip Attack] Evaluating ({desc})...")
    correct = 0
    total = 0
    asr_counts = torch.zeros(num_labels, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            batch_size = input_ids.size(0)

            kv_logits_list = []

            for i in range(batch_size):
                ids = input_ids[i: i + 1]
                mask = attention_mask[i: i + 1]

                seq_len = mask.sum(dim=1).item()
                last_idx = seq_len - 1

                # If only one token, fall back to full forward
                if last_idx == 0:
                    out_full = full_model(input_ids=ids, attention_mask=mask)
                    kv_logits_list.append(out_full.logits)
                    continue

                prefix_ids = ids[:, :last_idx]
                last_id = ids[:, last_idx: last_idx + 1]
                prefix_mask = mask[:, :last_idx]
                last_mask = mask[:, last_idx: last_idx + 1]

                prefix_outputs = base_model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = prefix_outputs.past_key_values

                corrupted_kv = corrupt_past_kv(past_kv, corruption_positions, corrupt_k=corrupt_k)
                extended_mask = torch.cat([prefix_mask, last_mask], dim=1)

                last_outputs = base_model(
                    input_ids=last_id,
                    attention_mask=extended_mask,
                    past_key_values=corrupted_kv,
                    use_cache=True,
                    return_dict=True,
                )

                hidden_last = last_outputs.last_hidden_state[:, -1, :]
                logits = classifier(hidden_last)
                kv_logits_list.append(logits)

            kv_logits = torch.cat(kv_logits_list, dim=0)

            if max_samples is not None:
                remaining = max_samples - total
                if remaining <= 0:
                    break
                if remaining < kv_logits.size(0):
                    kv_logits = kv_logits[:remaining]
                    labels = labels[:remaining]

            kv_predictions = kv_logits.argmax(dim=-1)
            correct += (kv_predictions == labels).sum().item()
            total += labels.size(0)

            for c in range(num_labels):
                asr_counts[c] += (kv_predictions == c).sum()

            if max_samples is not None and total >= max_samples:
                break

    if total == 0:
        return 0.0, np.zeros(num_labels, dtype=np.float32)

    acc = correct / total
    asr_per_label = (asr_counts.float() / total).cpu().numpy()

    print(f"[Bit-Flip Attack] Accuracy: {acc:.4f}")
    print("[Bit-Flip Attack] ASR per class:")
    for c in range(num_labels):
        print(f"  Class {c}: {asr_per_label[c]:.4f}")
    


    return acc, asr_per_label


def evaluate_baseline(valid_dataloader, base_model, classifier, full_model, device):
    """
    Baseline KV evaluation (no corruption).
    """
    print("\n[Baseline] Evaluating prefix-suffix KV accuracy...")
    correct_kv = 0
    total_kv = 0

    with torch.no_grad():
        #for batch in tqdm(valid_dataloader, desc="Baseline Eval", leave=False):
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            kv_logits_list = []

            for i in range(input_ids.size(0)):
                ids = input_ids[i: i + 1]
                mask = attention_mask[i: i + 1]

                seq_len = mask.sum(dim=1).item()
                last_idx = seq_len - 1

                if last_idx == 0:
                    out_full = full_model(input_ids=ids, attention_mask=mask)
                    kv_logits_list.append(out_full.logits)
                    continue

                prefix_ids = ids[:, :last_idx]
                last_id = ids[:, last_idx: last_idx + 1]
                prefix_mask = mask[:, :last_idx]
                last_mask = mask[:, last_idx: last_idx + 1]

                prefix_outputs = base_model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = prefix_outputs.past_key_values
                extended_mask = torch.cat([prefix_mask, last_mask], dim=1)

                last_outputs = base_model(
                    input_ids=last_id,
                    attention_mask=extended_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_last = last_outputs.last_hidden_state[:, -1, :]
                logits = classifier(hidden_last)
                kv_logits_list.append(logits)

            kv_logits = torch.cat(kv_logits_list, dim=0)
            kv_predictions = kv_logits.argmax(dim=-1)
            correct_kv += (kv_predictions == labels).sum().item()
            total_kv += labels.size(0)

    kv_acc = correct_kv / total_kv if total_kv > 0 else 0.0
    print(f"[Baseline] Accuracy: {kv_acc:.4f}")
    return kv_acc


# ======================================
#   Multi-class one-bit search
# ======================================

def select_positions_all_classes(
    calib_dataloader,
    base_model,
    classifier,
    full_model,
    device,
    num_labels: int,
    n_calib_samples: int,
    top_k: int,
    candidate_layers=None,
    top_bottom: str = "top",
    max_eval_samples=None,
    threshold: float = 0.99,
    corrupt_k: int = 1,
    top_m_per_class: int = 1,
):
    """
        For each candidate layer:
      1) Use training data to collect KV and rank positions in that layer.
      2) Take the top_k positions (by L2 importance) for that layer.
      3) For each of these positions (ONE BIT AT A TIME):
           - Flip just that single bit (single position)
           - Evaluate ASR for all classes on calib_dataloader
            4) For each class c, keep top_m_per_class best (layer, position) by ASR[c].

    Early-stop logic:
            - Stop when every class has top_m_per_class positions and every stored
                ASR is >= threshold.

    Returns:
        best_layer_per_label: dict[label_id] -> layer_idx
        best_positions_per_label: dict[label_id] -> list[ position_dict ]
        best_asr_per_label: np.ndarray[num_labels]
        layer_stats: dict[(layer_idx, pos_index)] -> { "acc": float, "asr_per_label": np.ndarray }
    """
    # Determine layers if not given
    if candidate_layers is None:
        layer_attr = getattr(base_model, "layers", None)
        if layer_attr is not None:
            candidate_layers = list(range(len(layer_attr)))
        else:
            num_hidden_layers = getattr(getattr(base_model, "config", None), "num_hidden_layers", 0)
            candidate_layers = list(range(num_hidden_layers))

    if not candidate_layers:
        raise RuntimeError("Could not determine model layers for auto selection.")

    if top_m_per_class < 1:
        raise ValueError("top_m_per_class must be >= 1")

    print("\n[Multi-Class One-Bit Search] Candidate layers:", candidate_layers)
    print(f"[Multi-Class One-Bit Search] Keeping top-{top_m_per_class} positions per class.")

    # For each class, track top-M candidates by ASR
    best_asr_per_label = np.full(num_labels, -1.0, dtype=np.float32)
    best_positions_per_label = {c: [] for c in range(num_labels)}
    best_layer_per_label = {c: None for c in range(num_labels)}

    # Optional diagnostics: per (layer, candidate_index)
    layer_stats = {}

    for layer_idx in candidate_layers:
        print(f"\n[Multi-Class One-Bit Search] Calibrating layer {layer_idx}")

        # Get top_k candidate positions in this layer
        positions = calibrate_positions_for_layer(
            calib_dataloader,
            base_model,
            device,
            n_calib_samples=n_calib_samples,
            top_k=top_k,
            layers=[layer_idx],
            top_bottom=top_bottom,
            corrupt_k=corrupt_k,
        )
        if not positions:
            print(f"[Multi-Class One-Bit Search] No positions for layer {layer_idx}, skipping.")
            continue

        # Try each candidate position ONE AT A TIME
        for idx, pos in enumerate(positions):
            print(
                f"\n[Multi-Class One-Bit Search] Evaluating single-bit candidate "
                f"{idx + 1}/{len(positions)} at layer {layer_idx}: "
                f"head={pos['head']}, dim={pos['dim']}, score={pos['score']:.4f}"
            )

            single_pos_list = [pos]  # pass as list so eval fn stays general

            acc, asr_vec = evaluate_attack_all_classes(
                calib_dataloader,
                base_model,
                classifier,
                full_model,
                device,
                single_pos_list,
                num_labels=num_labels,
                corrupt_k=corrupt_k,
                max_samples=max_eval_samples,
                desc=f"Calibration Attack Eval (layer {layer_idx}, pos {idx})",
            )

            layer_stats[(layer_idx, idx)] = {"acc": acc, "asr_per_label": asr_vec}

            # Update top-M positions per class
            for c in range(num_labels):
                class_candidates = best_positions_per_label[c]

                candidate_with_asr = {
                    "layer": pos["layer"],
                    "head": pos["head"],
                    "dim": pos["dim"],
                    "score": pos.get("score", 0.0),
                    "calib_asr": float(asr_vec[c]),
                }

                duplicate_idx = next(
                    (
                        i for i, existing in enumerate(class_candidates)
                        if existing["layer"] == candidate_with_asr["layer"]
                        and existing["head"] == candidate_with_asr["head"]
                        and existing["dim"] == candidate_with_asr["dim"]
                    ),
                    None,
                )

                if duplicate_idx is not None:
                    if candidate_with_asr["calib_asr"] > class_candidates[duplicate_idx]["calib_asr"]:
                        class_candidates[duplicate_idx] = candidate_with_asr
                elif len(class_candidates) < top_m_per_class:
                    class_candidates.append(candidate_with_asr)
                else:
                    min_idx = min(range(len(class_candidates)), key=lambda i: class_candidates[i]["calib_asr"])
                    if candidate_with_asr["calib_asr"] > class_candidates[min_idx]["calib_asr"]:
                        class_candidates[min_idx] = candidate_with_asr

                class_candidates.sort(key=lambda x: x["calib_asr"], reverse=True)

                if class_candidates:
                    best_asr_per_label[c] = class_candidates[0]["calib_asr"]
                    best_layer_per_label[c] = class_candidates[0]["layer"]

            # print the best ASR per class so far
            print("\n[Multi-Class One-Bit Search] Top candidates per class so far:")
            for c in range(num_labels):
                candidates = best_positions_per_label[c]
                if not candidates:
                    print(f"  Class {c}: no candidate bit found yet.")
                else:
                    top_items = ", ".join(
                        [
                            f"(layer={cand['layer']}, head={cand['head']}, dim={cand['dim']}, ASR={cand['calib_asr']:.4f})"
                            for cand in candidates
                        ]
                    )
                    print(f"  Class {c}: {top_items}")

            # ===== EARLY STOP CHECK HERE =====
            # Stop if each class has M candidates and all these candidates meet threshold.
            all_classes_ready = True
            for c in range(num_labels):
                candidates = best_positions_per_label[c]
                if len(candidates) < top_m_per_class:
                    all_classes_ready = False
                    break
                if any(cand["calib_asr"] < threshold for cand in candidates):
                    all_classes_ready = False
                    break

            if all_classes_ready:
                print(
                    "\n[Multi-Class One-Bit Search] Early stopping: "
                    f"all classes reached top-{top_m_per_class} candidates with ASR >= {threshold:.4f}."
                )
                return best_layer_per_label, best_positions_per_label, best_asr_per_label, layer_stats

    print("\n[Multi-Class One-Bit Search] Summary (top candidates per class):")
    for c in range(num_labels):
        candidates = best_positions_per_label[c]
        if not candidates:
            print(f"  Class {c}: no candidate bit found.")
        else:
            top_items = ", ".join(
                [
                    f"(layer={cand['layer']}, head={cand['head']}, dim={cand['dim']}, ASR={cand['calib_asr']:.4f})"
                    for cand in candidates
                ]
            )
            print(f"  Class {c}: {top_items}")

    return best_layer_per_label, best_positions_per_label, best_asr_per_label, layer_stats


# ======================================
#   Save/Load flip locations
# ======================================

def save_flip_locations(filepath, best_layer_per_label, best_positions_per_label, best_asr_per_label, num_labels):
    """
    Save the flip locations for each class to a JSON file.
    
    Args:
        filepath: Path to save the JSON file
        best_layer_per_label: dict[label_id] -> layer_idx
        best_positions_per_label: dict[label_id] -> list[position_dict]
        best_asr_per_label: np.ndarray[num_labels]
        num_labels: Number of classes
    """
    flip_data = {
        "num_labels": num_labels,
        "flip_locations": {}
    }
    
    for c in range(num_labels):
        layer = best_layer_per_label[c]
        asr = float(best_asr_per_label[c]) if isinstance(best_asr_per_label[c], (np.floating, float)) else best_asr_per_label[c]
        pos_list = best_positions_per_label[c]
        
        if not pos_list or layer is None:
            flip_data["flip_locations"][str(c)] = None
        else:
            serialized_list = []
            for pos in pos_list:
                serialized_list.append(
                    {
                        "layer": int(pos.get("layer", layer)),
                        "head": int(pos["head"]),
                        "dim": int(pos["dim"]),
                        "calib_asr": float(pos.get("calib_asr", asr)),
                        "score": float(pos.get("score", 0.0)),
                    }
                )
            flip_data["flip_locations"][str(c)] = serialized_list
    
    with open(filepath, 'w') as f:
        json.dump(flip_data, f, indent=2)
    
    print(f"\n[Save] Flip locations saved to: {filepath}")


def load_flip_locations(filepath):
    """
    Load flip locations from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        best_layer_per_label: dict[label_id] -> layer_idx
        best_positions_per_label: dict[label_id] -> list[position_dict]
        best_asr_per_label: np.ndarray[num_labels]
        num_labels: Number of classes
    """
    with open(filepath, 'r') as f:
        flip_data = json.load(f)
    
    num_labels = flip_data["num_labels"]
    best_layer_per_label = {}
    best_positions_per_label = {}
    best_asr_per_label = np.zeros(num_labels, dtype=np.float32)
    
    for c in range(num_labels):
        flip_info = flip_data["flip_locations"].get(str(c))
        
        if flip_info is None:
            best_layer_per_label[c] = None
            best_positions_per_label[c] = []
            best_asr_per_label[c] = -1.0
        else:
            # Backward compatibility: older format stored a single dict per class.
            if isinstance(flip_info, dict):
                candidates = [flip_info]
            elif isinstance(flip_info, list):
                candidates = flip_info
            else:
                candidates = []

            normalized_candidates = []
            for cand in candidates:
                normalized_candidates.append(
                    {
                        "layer": cand["layer"],
                        "head": cand["head"],
                        "dim": cand["dim"],
                        "score": cand.get("score", 0.0),
                        "calib_asr": cand.get("calib_asr", -1.0),
                    }
                )

            normalized_candidates.sort(key=lambda x: x.get("calib_asr", -1.0), reverse=True)
            best_positions_per_label[c] = normalized_candidates

            if normalized_candidates:
                best_layer_per_label[c] = normalized_candidates[0]["layer"]
                best_asr_per_label[c] = normalized_candidates[0].get("calib_asr", -1.0)
            else:
                best_layer_per_label[c] = None
                best_asr_per_label[c] = -1.0
    
    print(f"\n[Load] Flip locations loaded from: {filepath}")
    print("[Load] Loaded flip locations per class:")
    for c in range(num_labels):
        if best_layer_per_label[c] is None:
            print(f"  Class {c}: No flip location")
        else:
            display = ", ".join(
                [
                    f"(layer={pos['layer']}, head={pos['head']}, dim={pos['dim']}, calib_asr={pos.get('calib_asr', -1.0):.4f})"
                    for pos in best_positions_per_label[c]
                ]
            )
            print(f"  Class {c}: {display}")
    
    return best_layer_per_label, best_positions_per_label, best_asr_per_label, num_labels


# ======================================
#   Main
# ======================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KV-cache bit-flip attack on classifiers (multi-class, single-bit search)."
    )

    # Model / dataset
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Model short name used during training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset the model was trained on.",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        required=True,
        choices=DATASETS + ["same"],
        help="Dataset used for calibration.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset_all",
        help="Root folder for the JSON datasets.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Defaults to ./TrainedModels/merged_<model>_<dataset>.",
    )
    parser.add_argument(
        "--threat_model",
        type=str,
        default="graybox",
        choices=["graybox", "whitebox"],
        help="Calibrate on the foundation backbone (graybox) or the victim (whitebox).",
    )

    # Calibration / attack configuration
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of candidate positions per layer. Each is tried as a single bit flip.",
    )
    parser.add_argument(
        "--top_bottom",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Use top or bottom positions by L2 magnitude.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=100,
        help="Number of calibration samples (prefix-only KV collection) per layer.",
    )
    parser.add_argument(
        "--calib_eval_samples",
        type=int,
        default=100,
        help="Max number of calibration samples used during ASR evaluation per candidate bit.",
    )
    parser.add_argument(
        "--candidate_layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of layer indices to search. If not set, all layers are used.",
    )
    parser.add_argument(
        "--top_m_per_class",
        type=int,
        default=3,
        help=(
            "Number of best flip positions to keep per class during calibration."
        ),
    )
    parser.add_argument(
        "--corrupt_k",
        type=int,
        default=1,
        help=(
            "Corrupt the n-k token of the prefix (1-based from the end). "
            "k=1 is the token immediately before the final-token pass."
        ),
    )

    # Dataloader / eval
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Dataloader batch size.",
    )
    parser.add_argument(
        "--max_attack_eval_samples",
        type=int,
        default=None,
        help="Optional cap on number of validation samples during final per-class attack eval.",
    )
    
    parser.add_argument(
        "--save_flip_locations",
        type=str,
        default=None,
        help="Path to save flip locations after calibration for quick inference later.",
    )
    
    parser.add_argument(
        "--load_flip_locations",
        type=str,
        default=None,
        help="Path to load pre-computed flip locations. If provided, skips calibration.",
    )
    # Model loading options
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=sorted(DTYPES),
        help="Victim dtype.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation. Omit for the transformers default.",
    )

    return parser.parse_args()


def load_model_and_components(model_path, model_kwargs, device, use_device_map=False):
    """
    Helper function to load model, extract base_model and classifier.
    
    Args:
        model_path: Path to the model checkpoint
        model_kwargs: Dictionary of kwargs for model loading
        device: Target device
        use_device_map: Whether device_map is used (for quantized models)
    
    Returns:
        model, base_model, classifier
    """
    print(f"\n[Model Loading] Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    if not use_device_map:
        model.to(device)
        print(f"[Model Loading] Model loaded to device: {device}")
    else:
        print(f"[Model Loading] Model loaded with device_map=auto")
    
    # Base LM and classifier head
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("Expected model.model to exist (base transformer).")
    
    if hasattr(model, "score"):
        classifier = model.score
    elif hasattr(model, "classifier"):
        classifier = model.classifier
    else:
        raise RuntimeError("Could not find classifier head (expected .score or .classifier).")
    
    return model, base_model, classifier


def get_layer0_head1_v_matrix(base_model):
    """
    Return a CPU-cloned slice of layer-0, head-1 V projection weights.
    Shape: [head_dim, hidden_size]
    """
    layers = getattr(base_model, "layers", None)
    if layers is None or len(layers) == 0:
        return None

    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return None

    v_proj = getattr(attn, "v_proj", None)
    if v_proj is None or not hasattr(v_proj, "weight"):
        return None

    weight = v_proj.weight.detach().float().cpu().clone()
    num_heads = getattr(getattr(base_model, "config", None), "num_attention_heads", None)
    if num_heads is None or num_heads <= 1:
        return None

    head_dim = weight.shape[0] // num_heads
    head_idx = 1
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    if end > weight.shape[0]:
        return None

    return weight[start:end, :]


def main():
    args = parse_args()
    if args.corrupt_k < 1:
        raise ValueError("--corrupt_k must be >= 1")

    foundation_model_name = MODEL_REGISTRY[args.model]

    merged_model_path = args.model_path or f"./TrainedModels/merged_{args.model}_{args.dataset}"

    print(f"Victim model: {merged_model_path}")
    print(f"Threat model: {args.threat_model}")
    print(f"Dataset: {args.dataset} | Calibration dataset: {args.calib_dataset}")


    # --------------------------
    # Load tokenizer (use eval model path for tokenizer)
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

    tokenizer.padding_side = "right" ########

    # --------------------------
    # Prepare model loading arguments
    # --------------------------
    model_kwargs = {"torch_dtype": DTYPES[args.dtype]}
    print(f"[Model Loading] dtype: {args.dtype}")

    # Quantization options
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both --load_in_8bit and --load_in_4bit simultaneously.")
    
    use_device_map = False
    if args.load_in_8bit:
        print("[Model Loading] Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
        use_device_map = True
    elif args.load_in_4bit:
        print("[Model Loading] Using 4-bit quantization")
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
        use_device_map = True
    
    # Attention implementation
    if args.attn_implementation:
        print(f"[Model Loading] Using attention implementation: {args.attn_implementation}")
        model_kwargs["attn_implementation"] = args.attn_implementation
        if args.attn_implementation == "flash_attention_2" and args.dtype == "float32":
            raise ValueError("flash_attention_2 requires --dtype float16 or bfloat16.")
    
    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------
    # Load calibration model first
    # --------------------------
    model, base_model, classifier = load_model_and_components(
        merged_model_path, model_kwargs, device, use_device_map
    )

    swap_backbone = args.threat_model == "graybox" and not args.load_flip_locations

    if swap_backbone:
        print(f"[Model Loading] Replacing calibration backbone with: {foundation_model_name}")
        old_base_model = model.model
        model.model = None
        base_model = None
        del old_base_model
        torch.cuda.empty_cache()

        foundation_base_model = AutoModel.from_pretrained(foundation_model_name, **model_kwargs)
        foundation_base_model.eval()

        if not use_device_map:
            foundation_base_model.to(device)

        model.model = foundation_base_model
        base_model = model.model
        del foundation_base_model



    # --------------------------
    # Load formatted dataset
    # --------------------------
    train_dataset, valid_dataset, num_labels, label2id, id2label = prepare_datasets(
        args.dataset,
        tokenizer,
        data_root=args.data_root,
    )

    print("Sample from train dataset:")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # --------------------------
    # Print basic info
    # --------------------------
    print(f"num_labels: {num_labels}")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")
    print(
        f"Calibration config: calib_samples={args.calib_samples}, "
        f"calib_eval_samples={args.calib_eval_samples}, top_k={args.top_k}, "
        f"top_bottom={args.top_bottom}, corrupt_k={args.corrupt_k}, "
        f"top_m_per_class={args.top_m_per_class}"
    )

    # --------------------------
    # Multi-class one-bit selection (use train set as calibration section)
    # --------------------------
    
    # Check if we should load pre-computed flip locations
    if args.load_flip_locations:
        print(f"\n[Mode] Loading pre-computed flip locations from: {args.load_flip_locations}")
        best_layer_per_label, best_positions_per_label, best_asr_per_label, loaded_num_labels = load_flip_locations(
            args.load_flip_locations
        )
        if loaded_num_labels != num_labels:
            raise ValueError(
                f"Mismatch in number of labels: loaded {loaded_num_labels} but expected {num_labels}"
            )
        layer_stats = {}  # No layer stats when loading
    else:
        # Perform calibration
        print("\n[Mode] Performing calibration to find flip locations...")
        
        if args.calib_dataset == "same":
            calib_dataloader = train_dataloader
        else:
            print(f"Loading Different calibration dataset from Eval Dataset: {args.calib_dataset}")
            # Load calibration dataset
            calib_dataset, _, _, _, _ = prepare_datasets_calib(
                args.calib_dataset,
                args.dataset,
                tokenizer,
                data_root=args.data_root,
            )
            print("Sample from calibration dataset:")
            print(tokenizer.decode(calib_dataset[0]["input_ids"]))
            calib_dataloader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False)

        if args.corrupt_k < 3:
            thereshold_value = 0.985
            print(f"Using threshold {thereshold_value} for corrupt_k < 3 (more severe corruption)")
        elif args.corrupt_k == 3:
            thereshold_value = 0.96
            print(f"Using threshold {thereshold_value} for corrupt_k=3 (fourth-last token corruption)")

        else:
            thereshold_value = 0.9
            print(f"Using threshold {thereshold_value} for corrupt_k={args.corrupt_k} (less severe corruption)")

        best_layer_per_label, best_positions_per_label, best_asr_per_label, layer_stats = select_positions_all_classes(
            calib_dataloader=calib_dataloader,
            base_model=base_model,
            classifier=classifier,
            full_model=model,
            device=device,
            num_labels=num_labels,
            n_calib_samples=args.calib_samples,
            top_k=args.top_k,
            candidate_layers=args.candidate_layers,
            top_bottom=args.top_bottom,
            max_eval_samples=args.calib_eval_samples,
            threshold=thereshold_value,
            corrupt_k=args.corrupt_k,
            top_m_per_class=args.top_m_per_class,
        )
        
        # Save flip locations if requested
        if args.save_flip_locations:
            save_flip_locations(
                args.save_flip_locations,
                best_layer_per_label,
                best_positions_per_label,
                best_asr_per_label,
                num_labels
            )

    # --------------------------
    # Restore the victim model for evaluation
    # --------------------------
    if swap_backbone:
        print("\n" + "="*60)
        print("[Model Switch] Calibration complete. Loading evaluation model...")
        print("="*60)

        calib_v_matrix = get_layer0_head1_v_matrix(base_model)
        if calib_v_matrix is None:
            print("[Verification] Could not snapshot calibration layer0/head1 V matrix.")
        else:
            print("[Verification] Snapshotted calibration layer0/head1 V matrix.")
        
        # Clear calibration model from memory
        del model, base_model, classifier
        torch.cuda.empty_cache()
        
        # Load evaluation model
        model, base_model, classifier = load_model_and_components(
            merged_model_path, model_kwargs, device, use_device_map
        )

        eval_v_matrix = get_layer0_head1_v_matrix(base_model)
        if calib_v_matrix is None or eval_v_matrix is None:
            print("[Verification] V matrix comparison skipped (matrix unavailable).")
        else:
            exactly_same = torch.equal(calib_v_matrix, eval_v_matrix)
            print(
                "[Verification] layer0/head1 V matrix is "
                f"{'SAME' if exactly_same else 'DIFFERENT'} between calibration and evaluation models."
            )
        print("[Model Switch] Evaluation model loaded successfully\n")

    print("\n[Final Evaluation] Per-class attack results on validation set")

    final_results = {}  # store everything for summary

    for c in range(num_labels):
        positions = best_positions_per_label[c]
        if not positions:
            print(f"  Class {c}: no candidate bit found, skipping.")
            continue

        print(f"\n[Final Evaluation] Target class {c} ({len(positions)} candidate positions)")

        class_candidate_results = []
        for cand_idx, pos in enumerate(positions):
            layer = pos["layer"]
            print(
                f"  -> Candidate {cand_idx + 1}/{len(positions)}: "
                f"layer={layer}, head={pos['head']}, dim={pos['dim']}, "
                f"calib_ASR={pos.get('calib_asr', -1.0):.4f}"
            )

            acc, asr_vec = evaluate_attack_all_classes(
                valid_dataloader,
                base_model,
                classifier,
                model,
                device,
                [pos],
                num_labels=num_labels,
                corrupt_k=args.corrupt_k,
                max_samples=args.max_attack_eval_samples,
                desc=f"Final Attack Eval (target class {c}, cand {cand_idx})",
            )

            class_candidate_results.append(
                {
                    "layer": layer,
                    "head": pos["head"],
                    "dim": pos["dim"],
                    "acc": acc,
                    "asr_target": asr_vec[c],
                    "calib_asr": pos.get("calib_asr", best_asr_per_label[c]),
                }
            )

        class_candidate_results.sort(key=lambda x: x["asr_target"], reverse=True)
        final_results[c] = class_candidate_results

    # --------------------------
    # Centralized Summary
    # --------------------------
    print("\n\n===== FINAL PER-CLASS SUMMARY =====")
    for c in range(num_labels):
        if c not in final_results:
            print(f"Class {c}: No attack position found.")
            continue

        avg_asr = float(np.mean([cand["asr_target"] for cand in final_results[c]]))
        r = final_results[c][0]
        print(
            f"Class {c}: "
            f"Accuracy={r['acc']:.4f} | "
            f"BestEvalASR={r['asr_target']:.4f} | "
            f"AvgEvalASR={avg_asr:.4f} | "
            f"Flip=(layer {r['layer']}, head {r['head']}, dim {r['dim']}) | "
            f"Calib_ASR={r['calib_asr']:.4f}"
        )
    print("===================================\n")


    # --------------------------
    # Baseline evaluation
    # --------------------------
    evaluate_baseline(valid_dataloader, base_model, classifier, model, device)


if __name__ == "__main__":
    main()




