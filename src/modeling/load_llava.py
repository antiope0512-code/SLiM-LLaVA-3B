# src/modeling/load_llava.py
import torch
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model import LlavaConfig
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def load_llava(model_path: str, clip_path: str):
    """
    Load LLaVA-3B + CLIP vision tower in a branch-tolerant way.
    Returns (tokenizer, model, image_processor).
    """
    cfg = LlavaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    image_processor = CLIPImageProcessor.from_pretrained(clip_path)

    # Try to ensure a proper vision tower is attached and on the right device
    backbone = getattr(model, "get_model", lambda: model)()
    if hasattr(backbone, "vision_tower_name"):
        backbone.vision_tower_name = clip_path

    if getattr(backbone, "vision_tower", None) is None:
        if hasattr(backbone, "initialize_vision_modules"):
            backbone.initialize_vision_modules(clip_path=clip_path)
        elif hasattr(backbone, "initialize_vision_tower"):
            backbone.initialize_vision_tower(clip_path)

    if hasattr(backbone, "vision_tower") and hasattr(backbone.vision_tower, "to"):
        backbone.vision_tower.to(next(model.parameters()).device)

    # Play nice with gradient checkpointing
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return tokenizer, model, image_processor
