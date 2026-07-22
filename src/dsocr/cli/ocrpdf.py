import argparse
from dsocr import dsocr_pdf
from dsocr.infer import parse_latex
from itertools import chain

#%%
def parse_page_nums(page_num_str, out=None):

    if page_num_str is None:
        return None
    
    out = out or []
    
    if ',' in page_num_str:
        page_nums = [parse_page_nums(num.strip(), out) for num in page_num_str.split(',')]
        out.extend(page_nums)

    elif '-' in page_num_str:
        start, end = map(int, page_num_str.split('-'))
        page_nums = list(range(start, end + 1))

        out.append(page_nums)
    else:
        out.append(int(page_num_str))
 
    return out


#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("-p", "--page_num", default=None, type=str, help="Page numbers to process")
    parser.add_argument('-o', '--output', default="output", help="Output directory")
    parser.add_argument('-s', '--saveas', default="result.md", help="Save the result as a specific filename")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for PDF to image conversion")

    args = parser.parse_args()

    if args.page_num is not None:
        if args.page_num.isnumeric():
            page_nums = [int(args.page_num)]
        else:
            page_nums = sorted(set(chain(*parse_page_nums(args.page_num))))
    else:
        page_nums = None
    
    dsocr_pdf(args.pdf, page_num=page_nums if args.page_num is not None else None, 
              output=args.output, dpi=args.dpi, save_path=args.saveas)



if __name__ == "__main__":
    main()