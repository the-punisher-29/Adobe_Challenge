import os
import glob
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, 'input')
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))

def extract_headings(pdf_path):
    outline = []
    font_sizes = []
    headings = []
    title = None
    # First pass: collect all font sizes
    for pagenum, page in enumerate(extract_pages(pdf_path)):
        for element in page:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if not hasattr(text_line, "__iter__"):
                        continue
                    for char in text_line:
                        if isinstance(char, LTChar):
                            font_sizes.append(char.size)
    if not font_sizes:
        return None
    # Heuristic: largest font size is title, next largest are headings
    unique_sizes = sorted(list(set(font_sizes)), reverse=True)
    if len(unique_sizes) == 1:
        # Only one font size, treat all as body
        return None
    # Map font size to heading level
    size_to_level = {}
    if len(unique_sizes) >= 3:
        size_to_level[unique_sizes[0]] = 'H1'
        size_to_level[unique_sizes[1]] = 'H2'
        size_to_level[unique_sizes[2]] = 'H3'
    elif len(unique_sizes) == 2:
        size_to_level[unique_sizes[0]] = 'H1'
        size_to_level[unique_sizes[1]] = 'H2'
    else:
        size_to_level[unique_sizes[0]] = 'H1'
    # Second pass: extract headings and title
    for pagenum, page in enumerate(extract_pages(pdf_path)):
        for element in page:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if not hasattr(text_line, "__iter__"):
                        continue
                    line_text = text_line.get_text().strip()
                    if not line_text:
                        continue
                    line_fonts = []
                    for char in text_line:
                        if isinstance(char, LTChar):
                            line_fonts.append(char.size)
                    if not line_fonts:
                        continue
                    from collections import Counter
                    most_common_size = Counter(line_fonts).most_common(1)[0][0]
                    if most_common_size in size_to_level:
                        level = size_to_level[most_common_size]
                        if level == 'H1' and title is None:
                            title = line_text
                        else:
                            headings.append({
                                'level': level,
                                'text': line_text,
                                'page': pagenum + 1
                            })
    if title is None and headings:
        title = headings[0]['text']
    return {
        'title': title if title else os.path.splitext(os.path.basename(pdf_path))[0],
        'outline': headings
    }

def process_all_pdfs():
    for pdf_path in pdf_files:
        result = extract_headings(pdf_path)
        if result:
            filename = os.path.splitext(os.path.basename(pdf_path))[0] + '_outline1.json'
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f'Extracted outline for {os.path.basename(pdf_path)} -> {output_path}')
        else:
            print(f'No headings found in {os.path.basename(pdf_path)}')

if __name__ == '__main__':
    process_all_pdfs() 