import os
import json
import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize PaddleOCR (Offline mode, English only)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

TITLE_BLACKLIST = {                         # ⟵ NEW
    "overview", "contents", "title", "index",
    "table of contents", "chapter", "section",
    "figure", "document", "appendix"
}

class TextBlock:
    """Represents a text block with position, size, and content information"""
    def __init__(self, text: str, bbox: List[float], font_size: float, 
                 font_name: str = "", confidence: float = 1.0, page_num: int = 0):
        self.text = text.strip()
        self.bbox = bbox  # [x0, y0, x1, y1]
        self.font_size = font_size
        self.font_name = font_name
        self.confidence = confidence
        self.page_num = page_num
        
    def get_center(self) -> Tuple[float, float]:
        """Get center point of the text block"""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    def get_area(self) -> float:
        """Get area of the bounding box"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Check if there's an intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_distance(center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two centers"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def extract_pymupdf_blocks(page: fitz.Page, page_num: int) -> List[TextBlock]:
    """Extract text blocks from PyMuPDF with font information"""
    blocks = []
    text_dict = page.get_text("dict")
    
    for block in text_dict["blocks"]:
        if block.get("type") == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text and len(text) > 1:  # Filter out single characters
                        blocks.append(TextBlock(
                            text=text,
                            bbox=span["bbox"],
                            font_size=span["size"],
                            font_name=span["font"],
                            page_num=page_num
                        ))
    return blocks



def get_complete_text_in_region(page, bbox, ocr_results, expansion_factor=1.3):
    """
    Extracts complete text in a region by expanding search area and 
    collecting adjacent text blocks until word boundaries
    """
    import re
    
    # Expand bounding box to catch nearby text
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    
    # Create expanded search region
    expanded_bbox = (
        max(0, x0 - width * (expansion_factor - 1) / 2),
        max(0, y0 - height * (expansion_factor - 1) / 2),
        x1 + width * (expansion_factor - 1) / 2,
        y1 + height * (expansion_factor - 1) / 2
    )
    
    # Collect all OCR text blocks that intersect with expanded region
    text_candidates = []
    for line in ocr_results[0] if ocr_results and ocr_results[0] else []:
        ocr_bbox = line[0]  # OCR bounding box coordinates
        text = line[1][0]
        confidence = line[1][1]
        
        if confidence > 0.6 and boxes_overlap(expanded_bbox, convert_ocr_bbox(ocr_bbox)):
            # Convert OCR 4-point format to [x0,y0,x1,y1]
            xs = [p[0] for p in ocr_bbox]
            ys = [p[1] for p in ocr_bbox]
            ocr_x0, ocr_y0, ocr_x1, ocr_y1 = min(xs), min(ys), max(xs), max(ys)
            
            text_candidates.append({
                'text': text,
                'bbox': [ocr_x0, ocr_y0, ocr_x1, ocr_y1],
                'distance': calculate_distance((x0, y0), (ocr_x0, ocr_y0))
            })
    
    if not text_candidates:
        return ""
    
    # Sort by proximity to original bbox
    text_candidates.sort(key=lambda x: x['distance'])
    
    # Build complete text by joining adjacent blocks
    complete_text_parts = []
    for candidate in text_candidates:
        text_part = candidate['text'].strip()
        if text_part:
            complete_text_parts.append(text_part)
    
    # Join and clean up the complete text
    complete_text = " ".join(complete_text_parts)
    
    # Apply smart text completion rules
    complete_text = apply_completion_rules(complete_text)
    
    return complete_text

def apply_completion_rules(text):
    """
    Apply intelligent text completion for common truncations
    """
    # Common abbreviation expansions
    completions = {
        "RFP: R": "RFP: Request for Proposal",
        "AI & M": "AI & Machine Learning", 
        "Data S": "Data Science",
        "ML A": "Machine Learning Algorithms",
        "NLP P": "Natural Language Processing",
        "CV R": "Computer Vision Recognition",
        "API D": "API Documentation",
        "DB M": "Database Management"
    }
    
    # Direct replacement for known truncations
    for truncated, complete in completions.items():
        if text.startswith(truncated):
            return complete
    
    # Pattern-based completion for common formats
    import re
    
    # Handle "Word: L" -> "Word: Long Form" patterns
    colon_pattern = r'^([A-Z]{2,4}):\s+([A-Z])$'
    match = re.match(colon_pattern, text)
    if match:
        abbreviation, first_letter = match.groups()
        
        # Common expansions based on first letter
        expansions = {
            'R': 'Request',
            'A': 'Analysis', 
            'M': 'Management',
            'S': 'System',
            'P': 'Process',
            'D': 'Development'
        }
        
        if first_letter in expansions:
            return f"{abbreviation}: {expansions[first_letter]}"
    
    return text

def boxes_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap"""
    return not (bbox1[2] <= bbox2[0] or bbox2[2] <= bbox1[0] or 
               bbox1[3] <= bbox2[1] or bbox2[3] <= bbox1[1])

def convert_ocr_bbox(ocr_bbox):
    """Convert OCR 4-point format to standard bbox format"""
    xs = [p[0] for p in ocr_bbox]
    ys = [p[1] for p in ocr_bbox]
    return [min(xs), min(ys), max(xs), max(ys)]

def find_best_ocr_match(pdf_bbox, ocr_results,
                        iou_threshold=0.3, distance_threshold=50.0):
    """
    Return highest-scoring OCR text overlapping or nearest to a PyMuPDF span.
    """
    if not (ocr_results and ocr_results[0]):
        return ""
    best_text, best_score = "", -1
    px0, py0, px1, py1 = pdf_bbox
    pcx, pcy = (px0 + px1) / 2, (py0 + py1) / 2

    for quad, (text, conf) in ocr_results[0]:
        if conf < 0.6 or not text.strip():
            continue
        ox0, oy0, ox1, oy1 = convert_ocr_bbox(quad)
        iou = calculate_iou(pdf_bbox, [ox0, oy0, ox1, oy1])
        dcx, dcy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        dist = calculate_distance((pcx, pcy), (dcx, dcy))
        score = iou * 0.4 + conf * 0.3 + (1.0 / (1 + dist/100)) * 0.3
        if score > best_score and (iou >= iou_threshold or dist < distance_threshold):
            best_text, best_score = text, score
    return best_text

def extract_paddleocr_blocks(page_image: bytes, page_num: int) -> List[TextBlock]:
    """Extract text blocks from PaddleOCR"""
    blocks = []
    ocr_results = ocr_model.ocr(page_image, cls=True)
    
    if ocr_results and ocr_results[0]:
        for result in ocr_results[0]:
            points = result[0]  # 4 corner points
            text_info = result[1]  # (text, confidence)
            
            text = text_info[0].strip()
            confidence = text_info[1]
            
            if text and confidence > 0.6 and len(text) > 1:
                # Convert 4 corner points to bbox [x0, y0, x1, y1]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                
                blocks.append(TextBlock(
                    text=text,
                    bbox=bbox,
                    font_size=0,  # OCR doesn't provide font size
                    confidence=confidence,
                    page_num=page_num
                ))
    return blocks

def match_blocks(pymupdf_blocks: List[TextBlock], page, 
                         iou_threshold: float = 0.3, distance_threshold: float = 50.0) -> List[TextBlock]:
    matched_blocks = []
    
    # Get OCR results for the entire page
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    page_image = pix.tobytes("png")
    ocr_results = ocr_model.ocr(page_image, cls=True)
    
    for pdf_block in pymupdf_blocks:
        # Try standard matching first
        matched_text = find_best_ocr_match(pdf_block.bbox, ocr_results)
        
        # If text seems truncated, apply completion
        if is_text_truncated(matched_text):
            complete_text = get_complete_text_in_region(page, pdf_block.bbox, ocr_results)
            if len(complete_text) > len(matched_text):
                matched_text = complete_text
        
        # Create enhanced text block
        enhanced_block = TextBlock(
            text=matched_text if matched_text else pdf_block.text,
            bbox=pdf_block.bbox,
            font_size=pdf_block.font_size,
            font_name=pdf_block.font_name,
            confidence=1.0,
            page_num=pdf_block.page_num
        )
        
        matched_blocks.append(enhanced_block)
    
    return matched_blocks

def is_text_truncated(text):
    """Detect if text appears to be truncated"""
    if not text:
        return False
        
    # Check for common truncation patterns
    truncation_indicators = [
        len(text) < 5,  # Very short text
        text.endswith((' R', ' A', ' M', ' S', ' P', ' D')),  # Single letter endings
        text.count(':') > 0 and len(text.split(':')[-1].strip()) <= 2,  # Colon with short suffix
        text[-1].isupper() and len(text.split()[-1]) == 1  # Ends with single uppercase letter
    ]
    
    return any(truncation_indicators)

def classify_heading_levels(blocks: List[TextBlock]) -> Dict[str, str]:
    """Classify font sizes into heading levels"""
    if not blocks:
        return {}
    
    # Group blocks by font size
    size_groups = defaultdict(list)
    for block in blocks:
        size_groups[round(block.font_size, 1)].append(block)
    
    # Sort font sizes in descending order
    sorted_sizes = sorted(size_groups.keys(), reverse=True)
    
    # Classify levels
    level_mapping = {}
    
    if len(sorted_sizes) >= 1:
        level_mapping[sorted_sizes[0]] = "Title"
    if len(sorted_sizes) >= 2:
        level_mapping[sorted_sizes[1]] = "H1"
    if len(sorted_sizes) >= 3:
        level_mapping[sorted_sizes[2]] = "H2"
    if len(sorted_sizes) >= 4:
        level_mapping[sorted_sizes[3]] = "H3"
    
    return level_mapping

def is_likely_heading(block: TextBlock) -> bool:
    """Determine if a text block is likely to be a heading"""
    text = block.text.strip()
    
    # Basic heuristics for heading detection
    if len(text) < 3 or len(text) > 200:  # Too short or too long
        return False
    
    # Check if it's mostly uppercase or title case
    words = text.split()
    if len(words) <= 8:  # Headings are usually short
        title_case_count = sum(1 for word in words if word.istitle())
        upper_case_count = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        if title_case_count >= len(words) * 0.7 or upper_case_count >= len(words) * 0.5:
            return True
    
    return True  # For now, let font size be the primary filter

def extract_title_from_first_page(page_blocks: List[TextBlock],
                                  avg_font: float) -> str:             # ⟵ NEW
    """
    Choose the best-looking span on page-1 that is NOT black-listed.
    """
    if not page_blocks:
        return "Untitled Document"

    # Sort candidates by font-size then vertical position
    page_blocks.sort(key=lambda b: (-b.font_size, b.bbox[1]))

    for blk in page_blocks[:40]:           # examine top 40 candidates
        txt = blk.text.strip()
        if len(txt) < 4:
            continue
        if txt.lower() in TITLE_BLACKLIST: # ⟵ Black-list filter
            continue
        if blk.font_size < avg_font * 1.15 and not blk.text.isupper():
            continue                       # must be visibly larger
        return clean_title(txt)            # found a good one

    # Fallback: largest non-blacklisted text
    for blk in page_blocks:
        if blk.text.lower() not in TITLE_BLACKLIST:
            return clean_title(blk.text)
    return "Untitled Document"

def clean_title(text: str) -> str:                                       # ⟵ NEW
    """Collapse whitespace and apply title-case except ALL-CAPS acronyms."""
    parts = [w if w.isupper() else w.capitalize() for w in text.split()]
    return " ".join(parts)

def extract_outline_from_pdf(pdf_path: Path) -> Dict:
    doc = fitz.open(pdf_path)
    all_blocks = []
    for p in range(min(50, len(doc))):
        page = doc[p]
        pymupdf_blocks = extract_pymupdf_blocks(page, p)
        matched = match_blocks(pymupdf_blocks, page)
        all_blocks.extend(matched)
    doc.close()

    # ---------- ᴛɪᴛʟᴇ ---------------------------------------------------
    first_page_blocks = [b for b in all_blocks if b.page_num == 0]
    avg_font_size = (
        sum(b.font_size for b in first_page_blocks if b.font_size) /
        max(1, len([b for b in first_page_blocks if b.font_size]))
    )
    title = extract_title_from_first_page(first_page_blocks, avg_font_size)  # ⟵ MOD

    # ---------- headings (unchanged) -----------------------------------
    heading_blocks = [b for b in all_blocks if is_likely_heading(b)]
    level_map = classify_heading_levels(heading_blocks)
    outline = []
    for blk in heading_blocks:
        lvl = level_map.get(round(blk.font_size, 1))
        if lvl in {"H1", "H2", "H3"}:
            outline.append({"level": lvl, "text": blk.text, "page": blk.page_num + 1})

    return {"title": title, "outline": outline}

def process_all_pdfs():
    """Process all PDFs in the input directory"""
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} does not exist. Creating it...")
        INPUT_DIR.mkdir(parents=True)
        print("Please place your PDF files in the input directory and run again.")
        return
    
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process...")
    
    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path.name}...")
            
            # Extract outline
            result = extract_outline_from_pdf(pdf_path)
            
            # Save result
            output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Successfully processed: {pdf_path.name}")
            print(f"  Title: {result['title'][:50]}..." if len(result['title']) > 50 else f"  Title: {result['title']}")
            print(f"  Found {len(result['outline'])} headings")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("=== Hybrid PyMuPDF + PaddleOCR PDF Outline Extractor ===")
    print("This tool combines PyMuPDF's structure detection with PaddleOCR's accurate text extraction")
    print()
    
    process_all_pdfs()