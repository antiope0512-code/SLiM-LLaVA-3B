# src/train.py
"""
Minimal training pipeline:
LLaVA load -> SLiM (pre-quantize/restore, WANDA, LoRA) -> COCO caption fine-tuning.
"""

import torch
from transformers import TrainingArguments, Trainer, set_seed
from src.modeling.load_llava import load_llava
from src.modeling.slim_apply import apply_slim
from src.modeling.collator import MultimodalCollator
from src.data.preprocessing import load_coco_caption

def main():
    set_seed(42)

    # TODO: change these to your local paths/IDs
    base_dir = "/path/to/llava-3b"                 # LLaVA-3B text backbone
    clip_dir = "openai/clip-vit-large-patch14-336" # HF Hub id is fine

    tokenizer, model, image_processor = load_llava(base_dir, clip_dir)
    model = apply_slim(model, prune_ratio=0.2, quant_bits=4, group_size=128)

    # enable checkpointing (branch differences may require no arg)
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()

    # small sanity run
    train_ds = load_coco_caption(image_processor, tokenizer, split="val", n=500, max_length=128)
    collator = MultimodalCollator(tokenizer)

    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    args = TrainingArguments(
        output_dir="checkpoints/llava3b-slim-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,   # keep pixel_values
        dataloader_num_workers=4,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

    # train LoRA params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, None),
    )

    trainer.train()

    final_dir = "checkpoints/llava3b-slim-lora-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

if __name__ == "__main__":
    main()
