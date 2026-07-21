import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

#%%

def infer_images(image_files, output='output', cuda_device=None, 
                 prompt=None, base_size=1024, image_size=768, overwrite=False,
                 crop_mode=True, save_results=True, pbar=True):

    if isinstance(image_files, (str, Path)):
        image_files = [image_files]

    if any(not Path(image_file).is_file() for image_file in image_files):
        raise FileNotFoundError("One or more image files do not exist.")
    
    prompt = prompt or "<image>\n<|grounding|>Convert the document to markdown."
    cuda_device = str(cuda_device) if cuda_device is not None else '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    if not Path(output).is_dir():
        Path(output).mkdir(parents=True, exist_ok=True)

    model_name = 'deepseek-ai/DeepSeek-OCR-2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True, trust_remote_code=True)
    model = model.eval().cuda().to(torch.bfloat16)
    
    for image_file in tqdm(image_files, desc="Processing images", disable=not pbar):

        output_path = Path(output) / Path(image_file).stem
        output_path.mkdir(parents=True, exist_ok=True)
        if not overwrite and (output_path / 'result.mmd').exists():
            print(f"Skipping {image_file} as output already exists. Use overwrite=True to force reprocessing.")
            continue

        with open(output_path / 'log.txt', 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                model.infer(tokenizer, prompt=prompt, image_file=str(image_file), 
                            output_path=str(output_path), base_size=base_size, image_size=image_size, 
                            crop_mode=crop_mode, save_results=save_results)
            finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__



#%%