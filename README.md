# SLiM-LoRA for LLaVA (WIP)

Efficient fine-tuning of **LLaVA-3B** via a simple SLiM stack:
**(1) safe pre-quantize/restore → (2) WANDA pruning → (3) LoRA**.

> **Status:** Experimental. Interfaces can change anytime.

## Highlights
- Safe pre-quantize-then-restore on Linear weights (skip NaN/Inf; catch errors).
- WANDA pruning with conservative sparsity (default 0.2).
- LoRA on attention/MLP projections; train LoRA-only (freeze base).
- Minimal COCO caption fine-tuning and evaluation (BLEU/METEOR/CIDEr).
- Prompt trimming during decoding for fair scoring.

## Install
```bash
pip install -r requirements.txt
# LLaVA is recommended from source:
git clone https://github.com/haotian-liu/LLaVA
pip install -e ./LLaVA
