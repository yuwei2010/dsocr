import torch
import os
import sys
import logging
import transformers
import shutil
import tempfile
import regex as re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasurfer.lib_objects.markdown_object import MarkdownObject
from datasurfer import DataBay, DataPool
from datasurfer.lib_objects.pdf_object import PDFPagesObject
from datasurfer.datautils import is_sequence
transformers.logging.set_verbosity_error()

#%%

def parse_latex(s):
    s = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', s)
    s = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', s)
    return s
    # return s.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')
#%%

def dsocr_images(image_files, output='output', cuda_device=None, 
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
    model = AutoModel.from_pretrained(model_name, use_safetensors=True, trust_remote_code=True, attn_implementation="eager")
    model = model.eval().cuda().to(torch.bfloat16)

    pbar = tqdm(image_files, desc="Processing images", disable=not pbar)

    objs = []
    for image_file in pbar:

        pbar.set_postfix({"Current Image": Path(image_file).name})
        output_path = Path(output) / Path(image_file).stem
        output_path.mkdir(parents=True, exist_ok=True)

        if not overwrite and (output_path / 'result.mmd').exists():
            pbar.write(f"Skipping '{image_file}' as output already exists. Use overwrite=True to force reprocessing.")
            objs.append(MarkdownObject((output_path / 'result.mmd'), name=Path(image_file).stem))
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
        objs.append(MarkdownObject((output_path / 'result.mmd'), name=Path(image_file).stem))

    dp = DataPool(objs)
    return dp
        
#%%
def dsocr_pdf(fpdf, page_num=None, output='output', dpi=100, save_path='result.md', **kwargs):
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    
    obj = PDFPagesObject(fpdf)
    if page_num is None:
        page_nums = obj.page_nums
    else:
        page_nums = [page_num] if not is_sequence(page_num) else page_num

    output = Path(output)
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)

    imgs = []
    for page_num in page_nums:
        image_path = Path(tmp_dir) / f'page_{page_num:04d}.png'
        obj.page_to_image(page_num, str(image_path), dpi=dpi)
        imgs.append(str(image_path))

    
    dp = dsocr_images(imgs, output=output, **kwargs)

    mds = []

    for obj in dp:
        md = obj.get_text()
        if (obj.path.parent / 'images').is_dir():
            imgs = sorted((obj.path.parent / 'images').glob('*'))
            if imgs:
                dst_dir = output / 'images'
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    img_name = f"{obj.name}_{img.name}"
                    shutil.copy2(img, dst_dir / img_name)
                    md = md.replace(f'![](images/{img.name})', f'![](images/{img_name})')
        mds.append(md)


    md = parse_latex('\n\n'.join(mds))

    with open(f"{output}/{save_path}", 'w', encoding='utf-8') as f:
        f.write(md)
    return dp




#%%