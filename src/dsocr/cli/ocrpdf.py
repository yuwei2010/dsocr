"""Command-line interface for running DeepSeek-OCR on PDF documents.

This module parses user-supplied page-number specifications (e.g. ``"1,3-5,7``)
into a flat list of zero-based page indices and delegates the actual OCR work
to :func:`dsocr.dsocr_pdf`.
"""

import argparse
from itertools import chain

from dsocr import dsocr_pdf


#%%
def parse_page_nums(page_num_str, out=None):
    """Parse a page-number specification string into a list of integer lists.

    Supports two syntaxes that may be freely combined with commas:

    * **Single page** – ``"7"`` → ``[[7]]``
    * **Range** – ``"3-5"`` → ``[[3, 4, 5]]``
    * **Comma-separated mix** – ``"1,3-5,7"`` → ``[[1], [3, 4, 5], [7]]``

    Parameters
    ----------
    page_num_str : str | None
        The raw page specification string supplied on the command line.
    out : list | None
        Accumulator used during recursive calls.  Callers normally leave this
        as ``None`` so a fresh list is created internally.

    Returns
    -------
    list[list[int]] | None
        A list whose elements are themselves lists of integers, or ``None``
        when *page_num_str* is ``None``.
    """

    # No specification → process every page downstream.
    if page_num_str is None:
        return None

    out = out or []

    # Comma-separated entries are split and processed recursively.
    if ',' in page_num_str:
        page_nums = [parse_page_nums(num.strip(), out) for num in page_num_str.split(',')]
        out.extend(page_nums)

    # Range syntax "start-end" → inclusive integer range.
    elif '-' in page_num_str:
        start, end = map(int, page_num_str.split('-'))
        page_nums = list(range(start, end + 1))

        out.append(page_nums)
    # Bare integer → single page.
    else:
        out.append(int(page_num_str))

    return out


#%%
def main():
    """Entry point for the ``dsocr-ocrpdf`` console script.

    Builds an :class:`argparse.ArgumentParser`, converts the ``--page_num``
    string into a deduplicated, zero-based list of page indices, and forwards
    every option to :func:`dsocr.dsocr_pdf`.
    """

    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on a PDF file and export the result as Markdown."
    )
    parser.add_argument("pdf", help="Path to the PDF file to process.")
    parser.add_argument(
        "-p", "--page_num", default=None, type=str,
        help="Page numbers to process, e.g. '1', '3-5', or '1,3-5,7'. "
             "Defaults to all pages."
    )
    parser.add_argument(
        '-o', '--output', default="output",
        help="Output directory for intermediate per-page results. Default: 'output'."
    )
    parser.add_argument(
        '-s', '--saveas', default="result.md",
        help="Filename for the final combined Markdown file. Default: 'result.md'."
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI used when rasterising PDF pages to images. Default: 300."
    )

    args = parser.parse_args()

    # Normalise the page specification into a flat, sorted, unique list.
    if args.page_num is not None:
        if args.page_num.isnumeric():
            # Single numeric page – wrap directly.
            page_nums = [int(args.page_num)]
        else:
            # Flatten the nested lists returned by parse_page_nums, then
            # deduplicate and sort.
            page_nums = sorted(set(chain(*parse_page_nums(args.page_num))))
    else:
        page_nums = None

    # Convert from 1-based (user-facing) to 0-based (internal) page indices.
    page_nums = [int(num) - 1 for num in page_nums]

    # Delegate to the core OCR pipeline.
    dsocr_pdf(
        args.pdf,
        page_num=page_nums if args.page_num is not None else None,
        output=args.output,
        dpi=args.dpi,
        save_path=args.saveas,
    )


if __name__ == "__main__":
    main()