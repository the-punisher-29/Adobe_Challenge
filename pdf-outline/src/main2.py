# main.py
import os
import json
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict

# Optional: load ML model if needed (dummy classifier here)
def dummy_heading_classifier(text: str, font_size: float, is_bold: bool) -> str:
    # Replace with ML model inference if available
    if font_size >= 18:
        return "H1"
    elif font_size >= 16:
        return "H2"
    elif is_bold and font_size >= 14:
        return "H3"
    return "BODY"

# Multilingual heading patterns
HEADING_PATTERNS = [
    r"^\d+(\.\d+){0,2}\\s+",              # 1., 1.1, 1.1.1
    r"^(Chapter|Section)\\s+\d+",            # English
    r"^第[一二三四五六七八九十百]+章",            # Chinese/Japanese: 第三章
    r"^第\d+節",                               # Japanese: 第2節
]


# Font-style-based heading detection fallback
class FontStyleHeuristic:
    def __init__(self):
        self.font_stats = {}

    def update(self, font: str, size: float):
        key = (font, size)
        self.font_stats[key] = self.font_stats.get(key, 0) + 1

    def get_body_font_size(self):
        sorted_fonts = sorted(self.font_stats.items(), key=lambda x: x[1], reverse=True)
        return sorted_fonts[0][0][1] if sorted_fonts else 12.0


def extract_outline(pdf_path: str) -> Dict:
    doc = fitz.open(pdf_path)
    fs = FontStyleHeuristic()
    outline = []
    title = ""

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    text = s["text"].strip()
                    if not text or len(text) < 3:
                        continue

                    font = s["font"]
                    size = s["size"]
                    is_bold = "Bold" in font or "bold" in font

                    fs.update(font, size)

                    # Apply ML or heuristic heading classifier
                    heading_type = dummy_heading_classifier(text, size, is_bold)

                    # Use regex fallback if classifier is BODY
                    if heading_type == "BODY":
                        for pattern in HEADING_PATTERNS:
                            if re.match(pattern, text):
                                heading_type = "H2"  # Assume H2 for structured patterns
                                break

                    if heading_type.startswith("H"):
                        outline.append({
                            "level": heading_type,
                            "text": text,
                            "page": page_num
                        })

                    # Guess title on first page
                    if not title and page_num == 1 and size >= 20:
                        title = text

    if not title:
        title = Path(pdf_path).stem

    return {
        "title": title,
        "outline": outline
    }


def process_all_pdfs(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    for file in pdf_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, Path(file).stem + ".json")

        try:
            result = extract_outline(input_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Processed {file} → {output_path}")
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = f"{base_dir}/input"
    OUTPUT_DIR = f"{base_dir}/output"
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR)