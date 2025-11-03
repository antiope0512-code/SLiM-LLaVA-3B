# src/modeling/collator.py
from torch.utils.data._utils.collate import default_collate

class MultimodalCollator:
    """
    Pads text via tokenizer, masks pad tokens in labels to -100,
    and stacks pixel_values into (B, 3, H, W).
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # ignore loss on padding
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        batch["pixel_values"] = default_collate(pixel_values)
        return batch
