import argparse
from dsocr import dsocr_images


def main():
    """Entry point for the ``dsocr-ocrimage`` console script.

    Builds an :class:`argparse.ArgumentParser` and forwards every option to
    :func:`dsocr.dsocr_images`.
    """

    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on an image file and export the result as Markdown."
    )
    parser.add_argument("image", help="Path to the image file to process.")
    parser.add_argument(
        '-o', '--output', default="output",
        help="Output directory for intermediate per-page results. Default: 'output'."
    )

    args = parser.parse_args()

    dsocr_images([args.image], output=args.output, pbar=False)

if __name__ == "__main__":
    main()