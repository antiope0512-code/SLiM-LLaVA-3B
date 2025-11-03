# src/eval.py
"""
COCO caption eval with BLEU/METEOR/CIDEr.
We trim the prompt from decoding before scoring.
"""

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from src.modeling.load_llava import load_llava

PROMPT = "<image>\nUSER: Describe the image.\nASSISTANT: "

def evaluate(model, tokenizer, image_processor, n=200, max_new_tokens=50):
    ds = load_dataset("lmms-lab/COCO-Caption2017", split=f"val[:{n}]")
    model.eval()

    refs_all, hyps_all, meteor_scores = [], [], []
    gts, res = {}, {}
    cider_eval = Cider()

    with torch.no_grad():
        for i, d in enumerate(ds):
            inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
            pixel_values = image_processor(d["image"], return_tensors="pt").pixel_values.to(model.device)

            out = model.generate(**inputs, pixel_values=pixel_values, max_new_tokens=max_new_tokens, do_sample=False)
            gen_ids = out[0][inputs["input_ids"].shape[1]:]  # drop prompt tokens
            caption = tokenizer.decode(gen_ids, skip_special_tokens=True)

            refs = [d["caption"]]
            refs_all.append([r.split() for r in refs])
            hyps_all.append(caption.split())

            meteor_scores.append(meteor_score(refs, caption))
            gts[str(i)] = refs
            res[str(i)] = [caption]

    bleu = corpus_bleu(refs_all, hyps_all, smoothing_function=SmoothingFunction().method1)
    meteor_avg = sum(meteor_scores) / max(1, len(meteor_scores))
    cider, _ = cider_eval.compute_score(gts, res)
    return {"BLEU": bleu, "METEOR": meteor_avg, "CIDEr": cider}

if __name__ == "__main__":
    base_dir = "/path/to/llava-3b"
    clip_dir = "openai/clip-vit-large-patch14-336"
    tokenizer, model, image_processor = load_llava(base_dir, clip_dir)
    print(evaluate(model, tokenizer, image_processor))
