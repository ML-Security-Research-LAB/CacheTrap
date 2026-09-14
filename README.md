# CacheTrap

Repository for *"CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs"*,
IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2026.

[arXiv](https://arxiv.org/abs/2511.22681)

CacheTrap searches for a single bit flip in the KV cache of a fine-tuned
LLM classifier and measures the resulting attack success rate (ASR) per target class.

---

## 1. Setup

Python 3.10, CUDA 12.x.

```bash
conda create -n cachetrap python=3.10
conda activate cachetrap
pip install -r requirements.txt
```

**Hugging Face access.** `meta-llama/Llama-2-7b-chat-hf` and
`meta-llama/Llama-3.1-8B-Instruct` are gated. Accept the licences on the Hub, then:

```bash
huggingface-cli login
```

---

## 2. Training

One model / one dataset:

```bash
python train_model.py --model llama3_1_8b --dataset arc_easy
```

The checkpoint is written to
`TrainedModels/merged_<model>_<dataset>/`.

All models × all datasets:

```bash
bash scripts/train.sh
```

Edit `MODELS`, `DATASETS`, and `NUM_GPUS` at the top of the script to run a subset.
Logs go to `logs/training/`.

---

## 3. Attack

```bash
python attack.py --model llama3_1_8b --dataset arc_easy --calib_dataset openbookqa
```

`--dataset` is the dataset the victim was trained on; `--calib_dataset` is the data
available to the attacker. Following the paper, victims trained on OpenBookQA, TREC
or ARC-Challenge are calibrated with ARC-Easy, and victims trained on ARC-Easy are
calibrated with OpenBookQA. The two batch scripts encode exactly that split:

| Script | Victim datasets | Calibration |
|---|---|---|
| `scripts/attack_bundle1.sh` | openbookqa, trec, arc_challenge | arc_easy |
| `scripts/attack_bundle2.sh` | arc_easy | openbookqa |

```bash
bash scripts/attack_bundle1.sh
bash scripts/attack_bundle2.sh
```

Each ends by running `summarize_logs.py` over its own log directory.
Run `python attack.py --help` for the full option list.

### Outputs

- `logs/<run>/<model>_<dataset>.log` — full per-run output
- `logs/<run>/summary.csv` — one row per target class (baseline accuracy, accuracy
  under attack, ASR, flip location)
- `logs/<run>/summary.txt` — the raw summary blocks
- `flip_locations/*.json` — flip locations, if `--save_flip_locations` was passed

To re-evaluate without repeating the search:

```bash
python attack.py --model llama3_1_8b --dataset arc_easy --calib_dataset openbookqa \
    --load_flip_locations flip_locations/llama3_1_8b_arc_easy.json
```

---

## 4. Quick test with a released checkpoint

To try the attack without training anything, download the released
[`merged_llama3_1_8b_arc_easy`](https://drive.google.com/file/d/1WDohlyP_K4mGIqEtrNnDmV3E6HqEhVNj/view?usp=sharing) checkpoint and place it under `TrainedModels/`:

```
TrainedModels/
└── merged_llama3_1_8b_arc_easy/
    ├── config.json
    ├── model-*.safetensors
    ├── tokenizer.json
    └── ...
```

Then run the following command to attack:

```bash
python attack.py --model llama3_1_8b --dataset arc_easy --calib_dataset openbookqa \
    --load_flip_locations flip_locations/llama3_1_8b_arc_easy.json
```

Alternatively, run the following for a fresh attack

```bash
python attack.py \
    --model llama3_1_8b \
    --dataset arc_easy \
    --calib_dataset openbookqa \
    --threat_model graybox \
    --save_flip_locations flip_locations/llama3_1_8b_arc_easy_fresh.json
```

This prints a per-class summary of baseline accuracy, accuracy under attack, ASR,
and the selected flip location.

---

## Citation

Cite the pre-print version as

```bibtex
@article{nahian2025cachetrap,
  title={CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs},
  author={Nahian, Mohaiminul Al and Almalky, Abeer Matar A and Aragonda, Gamana and Zhou, Ranyang and Ahmed, Sabbir and Ponomarev, Dmitry and Yang, Li and Angizi, Shaahin and Rakin, Adnan Siraj},
  journal={arXiv preprint arXiv:2511.22681},
  year={2025}
}
```
