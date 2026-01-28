from prompt import prompt
import os
import PIL
import PIL.Image
from tqdm import tqdm
import torch
import numpy as np
import random
import json
import re
def load_labels(
        label_file: str
    ):
    label_dict = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                image_name = parts[0]
                image_name_base = os.path.splitext(image_name)[0]
                label = parts[1]
                label_dict[image_name_base] = label
    return label_dict


def make_think_data_two_versions(
        label_file: str,
        image_dir: str,
        train_data_cot: str,
        disable_marker: str = "<|think|><|think|>", 
        disable_output_mode: str = "label_only"
    ):
    
    train_data = json.load(open(train_data_cot, "r", encoding="utf-8"))
    labels = load_labels(label_file)
    res = []

    for item in tqdm(train_data):
        img_path = os.path.join(image_dir, item["image_name"])
        image_key = os.path.splitext(item["image_name"])[0]

        if image_key not in labels:
            continue

        label = labels[image_key]
        cot = item.get("chain_of_thought", "")

        prompt_on = prompt
        prompt_off = f"{prompt}{disable_marker}"

        answer_on = f"<|think|>{cot}<|think|><|label|>{label}<|label|>"

        if disable_output_mode == "label_only":
            answer_off = f"<|label|>{label}<|label|>"
        elif disable_output_mode == "empty_think_plus_label":
            answer_off = f"<|think|><|think|><|label|>{label}<|label|>"
        else:
            raise ValueError("disable_output_mode must be 'label_only' or 'empty_think_plus_label'")

        try:
            
            with PIL.Image.open(img_path) as img:
                img_copy = img.copy()

            conv_on = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": prompt_on},
                        {"type": "image", "image": img_copy},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_on}],
                },
            ]
            res.append({"messages": conv_on})

            conv_off = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": prompt_off},
                        {"type": "image", "image": img_copy},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_off}],
                },
            ]
            res.append({"messages": conv_off})

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return res

def set_seed(
        seed: int=42
    ):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_tta(
        image: PIL.Image, 
        use_tta: bool = False
    ):
    augmented_images = []
    augmented_images.append(image)
    if use_tta:
        augmented_images.append(image.transpose(PIL.Image.FLIP_LEFT_RIGHT))
        augmented_images.append(image.transpose(PIL.Image.FLIP_TOP_BOTTOM))
        augmented_images.append(image.rotate(90))
        augmented_images.append(image.rotate(180))
        augmented_images.append(image.rotate(270))
    return augmented_images

def inference_with_tta(
        image: PIL.Image, 
        instruction: str, 
        model, 
        tokenizer, 
        use_tta: bool = False,
    ):
    augmented_images = apply_tta(image, use_tta)
    
    results = []
    
    for aug_image in augmented_images:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            aug_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False, 
            temperature=1.0,   
            top_p=1.0,        
            top_k=0,           
            num_beams=1       
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_text)
        output_text = output_text.split("assistant")[-1].strip()
        pattern = r'<\|label\|>(.*?)<\|label\|>'
        match = re.search(pattern, output_text, flags=re.DOTALL)
        res = match.group(1).strip() if match else None
        results.append(res)
    
    final_result = max(set(results), key=results.count)
    return final_result