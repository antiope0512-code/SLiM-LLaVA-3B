# src/modeling/slim_apply.py
import torch
import torch.nn as nn
from slim.quantization.quantization import Quantizer
from slim.prune import prune_wanda
from slim.lora import add_lora

@torch.no_grad()
def _safe_quantize(quantizer, W, num_bits=4):
    """
    Conservative pre-quantize-then-restore for Linear weights.
    Skips empty/non-finite tensors and catches runtime errors.
    """
    if W.numel() == 0:
        return W
    if not torch.isfinite(W.float()).all():
        return W
    try:
        Q = quantizer.quantize(W, num_bits=num_bits)
        return quantizer.dequantize_absmax(Q)
    except Exception:
        return W

def apply_slim(model, prune_ratio=0.2, quant_bits=4, group_size=128):
    """
    SLiM stack: (1) pre-quantize-restore Linear, (2) WANDA prune, (3) add LoRA.
    Trains LoRA-only by freezing base weights.
    """
    try:
        llm_layers = model.get_model().layers
    except AttributeError:
        llm_layers = model.model.layers

    # (1) pre-quantize-restore for Linear weights
    quantizer = Quantizer(matrix_type="weight", num_bits=quant_bits, group_size=group_size)
    visited = set()
    with torch.no_grad():
        for layer in llm_layers:
            for _, mod in layer.named_modules():
                if isinstance(mod, nn.Linear) and id(mod) not in visited:
                    mod.weight.copy_(_safe_quantize(quantizer, mod.weight, num_bits=quant_bits))
                    visited.add(id(mod))

    # (2) prune with WANDA (keep it conservative by default)
    prune_wanda(llm_layers, sparsity_ratio=prune_ratio)

    # (3) LoRA on attention/MLP projection layers
    add_lora(
        llm_layers,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        rank=8, alpha=16, dropout=0.05
    )

    # train LoRA only
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    return model
