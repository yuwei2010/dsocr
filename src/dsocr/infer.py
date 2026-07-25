"""
Inference module for DeepSeek-OCR-2.

This module provides functions to run optical character recognition (OCR) on
either standalone image files or PDF documents using the DeepSeek-OCR-2 model.
The recognized text is returned as Markdown, with LaTeX formulas normalized
and extracted images preserved alongside the output.
"""

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

# Suppress verbose transformer warnings so they don't clutter the log output.
transformers.logging.set_verbosity_error()

#%%
# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def parse_latex(s):
    """
    Normalize LaTeX delimiters in a markdown string.

    DeepSeek-OCR may emit ``\\[ ... \\]`` for display math and ``\\( ... \\)``
    for inline math.  These are converted to the more portable ``$$ ... $$``
    and ``$ ... $`` delimiters respectively.  Existing single ``$`` delimited
    expressions are also tidied by trimming surrounding whitespace.
    """
    # Convert display math: \[ ... \]  ->  $$ ... $$
    s = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', s)
    # Convert inline math: \( ... \)  ->  $ ... $
    s = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', s)
    # Normalize whitespace inside existing single-$ inline expressions.
    s = re.sub(r'\$\s*(.*?)\s*\$', lambda m: f'${m.group(1)}$', s)
    return s
    # return s.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')
#%%
# ---------------------------------------------------------------------------
# Image OCR
# ---------------------------------------------------------------------------

def dsocr_images(image_files, output='output', cuda_device=None, 
                 prompt=None, base_size=1024, image_size=768, overwrite=False,
                 crop_mode=True, save_results=True, pbar=True):
    """
    Run DeepSeek-OCR-2 on a list of image files.

    Parameters
    ----------
    image_files : str | Path | list[str | Path]
        One or more image file paths to process.
    output : str | Path
        Directory where per-image result folders are created.
    cuda_device : str | int | None
        CUDA device id to use (e.g. ``'0'``).  ``None`` defaults to ``'0'``.
    prompt : str | None
        Inference prompt sent to the model.  Defaults to a grounding prompt
        that asks the model to convert the document to markdown.
    base_size, image_size : int
        Model-side image processing parameters forwarded to ``model.infer``.
    overwrite : bool
        If ``False``, images whose output already exists are skipped.
    crop_mode : bool
        Whether the model should crop sub-regions of the image before recognition.
    save_results : bool
        Whether the model should persist intermediate results to disk.
    pbar : bool
        Show a progress bar.

    Returns
    -------
    DataPool
        A ``DataPool`` of ``MarkdownObject`` instances, one per processed image.
    """
    # Normalize a single path argument into a list for uniform handling.
    if isinstance(image_files, (str, Path)):
        image_files = [image_files]

    # Validate that every requested image actually exists on disk.
    if any(not Path(image_file).is_file() for image_file in image_files):
        raise FileNotFoundError("One or more image files do not exist.")
    
    # Default prompt instructs the model to convert the document to markdown.
    prompt = prompt or "<image>\n<|grounding|>Convert the document to markdown."
    # Pin the CUDA device before loading the model so it lands on the right GPU.
    cuda_device = str(cuda_device) if cuda_device is not None else '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Ensure the output root directory exists.
    if not Path(output).is_dir():
        Path(output).mkdir(parents=True, exist_ok=True)

    # Load the DeepSeek-OCR-2 tokenizer and model from the Hugging Face hub.
    model_name = 'deepseek-ai/DeepSeek-OCR-2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True, trust_remote_code=True, attn_implementation="eager")
    # Switch to eval mode, move to GPU, and cast to bfloat16 for inference.
    model = model.eval().cuda().to(torch.bfloat16)

    # Wrap the image list with a progress bar (disabled when pbar=False).
    pbar = tqdm(image_files, desc="Processing images", disable=not pbar)

    objs = []
    for image_file in pbar:

        # Update the progress bar with the file currently being processed.
        pbar.set_postfix({"Current Image": Path(image_file).name})
        # Each image gets its own sub-folder named after the image stem.
        output_path = Path(output) / Path(image_file).stem
        output_path.mkdir(parents=True, exist_ok=True)

        # Skip already-processed images unless the caller forces an overwrite.
        if not overwrite and (output_path / 'result.mmd').exists():
            pbar.write(f"Skipping '{image_file}' as output already exists. Use overwrite=True to force reprocessing.")
            objs.append(MarkdownObject((output_path / 'result.mmd'), name=Path(image_file).stem))
            continue

        # Redirect stdout/stderr into a per-image log file so that the model's
        # verbose console output is captured on disk rather than the terminal.
        with open(output_path / 'log.txt', 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                # Run the actual OCR inference via the model's built-in method.
                model.infer(tokenizer, prompt=prompt, image_file=str(image_file), 
                            output_path=str(output_path), base_size=base_size, image_size=image_size, 
                            crop_mode=crop_mode, save_results=save_results)
            finally:
                # Always restore the original streams, even if inference fails.
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
        # Wrap the resulting markdown file in a MarkdownObject for downstream use.
        objs.append(MarkdownObject((output_path / 'result.mmd'), name=Path(image_file).stem))

    # Bundle all per-image markdown objects into a single DataPool.
    dp = DataPool(objs)
    return dp
        
#%%
# ---------------------------------------------------------------------------
# PDF OCR
# ---------------------------------------------------------------------------

def dsocr_pdf(fpdf, page_num=None, output='output', dpi=100, save_path='result.md', **kwargs):
    """
    Run DeepSeek-OCR-2 on a PDF document.

    The PDF is first rasterized into per-page PNG images, which are then fed
    to :func:`dsocr_images`.  The per-page markdown results are concatenated
    into a single ``result.md`` file, and extracted images are copied into a
    shared ``images/`` folder with page-prefixed names to avoid collisions.

    Parameters
    ----------
    fpdf : str | Path
        Path to the input PDF file.
    page_num : int | sequence[int] | None
        A single page number, a sequence of page numbers, or ``None`` to
        process every page in the document.
    output : str | Path
        Directory for intermediate per-page OCR results.
    dpi : int
        Resolution used when rasterizing PDF pages to images.
    save_path : str | Path
        Path of the final combined markdown file.
    **kwargs
        Additional keyword arguments forwarded to :func:`dsocr_images`
        (e.g. ``cuda_device``, ``prompt``, ``overwrite``).

    Returns
    -------
    DataPool
        The ``DataPool`` returned by :func:`dsocr_images`.
    """
    # Create a temporary directory to hold the rasterized page images.
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    
    # Wrap the PDF in a datasurfer PDFPagesObject for page-level access.
    obj = PDFPagesObject(fpdf)
    # Determine which pages to process: all pages, a single page, or a subset.
    if page_num is None:
        page_nums = obj.page_nums
    else:
        page_nums = [page_num] if not is_sequence(page_num) else page_num

    # Ensure the intermediate output directory exists.
    output = Path(output)
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)

    # Rasterize each requested page to a PNG inside the temp directory.
    imgs = []
    for page_num in page_nums:
        image_path = Path(tmp_dir) / f'page_{page_num:04d}.png'
        obj.page_to_image(page_num, str(image_path), dpi=dpi)
        imgs.append(str(image_path))

    # The final markdown is written next to `save_path`; ensure its parent dir exists.
    root = Path(save_path).resolve().parent
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)

    # Run OCR over all rasterized page images.
    dp = dsocr_images(imgs, output=output, **kwargs)

    # Collect per-page markdown and relocate extracted images into one folder.
    mds = []
    for obj in dp:
        md = obj.get_text()
        # If the model extracted images for this page, copy them with a unique
        # page-prefixed name and rewrite the markdown references accordingly.
        if (obj.path.parent / 'images').is_dir():
            imgs = sorted((obj.path.parent / 'images').glob('*'))
            if imgs:
                dst_dir = root / 'images'
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    img_name = f"{obj.name}_{img.name}"
                    shutil.copy2(img, dst_dir / img_name)
                    md = md.replace(f'![](images/{img.name})', f'![](images/{img_name})')
        mds.append(parse_latex(md))


    # Concatenate all pages, normalize LaTeX delimiters, and write the result.
    md = '\n\n'.join(mds)
    Path(save_path).write_text(md, encoding='utf-8')

    return dp




#%%