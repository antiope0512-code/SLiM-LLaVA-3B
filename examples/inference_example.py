# examples/inference_example.py
from PIL import Image
from src.modeling.load_llava import load_llava

PROMPT = "<image>\nUSER: Describe the image.\nASSISTANT: "

if __name__ == "__main__":
    base_dir = "/path/to/llava-3b"
    clip_dir = "openai/clip-vit-large-patch14-336"

    tokenizer, model, image_processor = load_llava(base_dir, clip_dir)

    img = Image.open("examples/demo.jpg")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    pixel_values = image_processor(img, return_tensors="pt").pixel_values.to(model.device)

    out = model.generate(**inputs, pixel_values=pixel_values, max_new_tokens=64, do_sample=False)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))
