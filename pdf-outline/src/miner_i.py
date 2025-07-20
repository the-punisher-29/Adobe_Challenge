import os
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

# Heuristic for heading detection: font size, boldness, and position
def extract_title_and_headings(pdf_path, max_pages=50):
    headings = []
    title = None
    font_stats = {}
    page_titles = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path)):
        if page_num >= max_pages:
            break
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if not hasattr(text_line, "__iter__"): continue
                    line_text = text_line.get_text().strip()
                    if not line_text: continue
                    fonts = []
                    for char in text_line:
                        if isinstance(char, LTChar):
                            fonts.append((char.fontname, char.size))
                    if not fonts: continue
                    # Use the most common font size in the line
                    sizes = [size for _, size in fonts]
                    main_size = max(set(sizes), key=sizes.count)
                    font_stats.setdefault(main_size, 0)
                    font_stats[main_size] += 1
                    page_titles.append((main_size, line_text, page_num+1))
    # Guess title: largest font, first page, not all caps
    if page_titles:
        page_titles.sort(reverse=True)
        for size, text, page in page_titles:
            if len(text.split()) > 2 and not text.isupper() and page == 1:
                title = text
                break
        if not title:
            title = page_titles[0][1]
    # Now, extract headings by re-parsing and using font size hierarchy
    font_sizes = sorted(font_stats, key=font_stats.get, reverse=True)
    if len(font_sizes) < 3:
        # fallback: use unique sizes
        font_sizes = sorted(set([s for s,_,_ in page_titles]), reverse=True)
    size_to_level = {}
    if len(font_sizes) >= 3:
        size_to_level[font_sizes[0]] = 'H1'
        size_to_level[font_sizes[1]] = 'H2'
        size_to_level[font_sizes[2]] = 'H3'
    elif len(font_sizes) == 2:
        size_to_level[font_sizes[0]] = 'H1'
        size_to_level[font_sizes[1]] = 'H2'
    elif len(font_sizes) == 1:
        size_to_level[font_sizes[0]] = 'H1'
    for page_num, page_layout in enumerate(extract_pages(pdf_path)):
        if page_num >= max_pages:
            break
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if not hasattr(text_line, "__iter__"): continue
                    line_text = text_line.get_text().strip()
                    if not line_text: continue
                    fonts = []
                    for char in text_line:
                        if isinstance(char, LTChar):
                            fonts.append((char.fontname, char.size))
                    if not fonts: continue
                    sizes = [size for _, size in fonts]
                    main_size = max(set(sizes), key=sizes.count)
                    level = size_to_level.get(main_size)
                    if level and len(line_text) < 120 and not line_text.islower():
                        headings.append({"level": level, "text": line_text, "page": page_num+1})
    return {"title": title or "", "outline": headings}

if __name__ == "__main__":
    import time
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + '_outline.json')
        print(f'Processing {pdf_file}...')
        start_time = time.time()
        result = extract_title_and_headings(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - start_time
        print(f'Processed {pdf_file} in {elapsed:.2f} seconds. Output: {output_path}')
