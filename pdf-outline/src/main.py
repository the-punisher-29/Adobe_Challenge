import os
import json
import fitz
import re
from collections import defaultdict
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PDFOutlineExtractor:
    def __init__(self):
        self.input_dir = Path(f"{base_dir}/input")
        self.output_dir = Path(f"{base_dir}/output")

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        
    def extract_outline(self, pdf_path):
        """Extract structured outline from PDF"""
        doc = fitz.open(pdf_path)
        
        # Collect all text spans with font information
        all_spans = []
        page_texts = {}
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            page_spans = []
            
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if len(text) > 1:  # Filter out single characters
                            span_info = {
                                "text": text,
                                "size": round(span["size"], 1),
                                "font": span["font"],
                                "flags": span["flags"],
                                "page": page_num,
                                "bbox": span["bbox"]
                            }
                            all_spans.append(span_info)
                            page_spans.append(span_info)
            
            page_texts[page_num] = page_spans
        
        doc.close()
        
        # Extract title (largest text on first page)
        title = self._extract_title(page_texts.get(1, []))
        
        # Extract headings
        outline = self._extract_headings(all_spans)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_title(self, first_page_spans):
        """Extract document title from first page"""
        if not first_page_spans:
            return "Untitled Document"
        
        # Find the largest font size on first page
        max_size = max(span["size"] for span in first_page_spans)
        
        # Get text with maximum font size
        title_candidates = [
            span["text"] for span in first_page_spans 
            if span["size"] == max_size and len(span["text"]) > 3
        ]
        
        if title_candidates:
            # Choose the longest candidate (likely the full title)
            return max(title_candidates, key=len)
        
        return "Untitled Document"
    
    def _extract_headings(self, all_spans):
        """Extract hierarchical headings from all spans"""
        # Filter potential headings based on font characteristics
        heading_candidates = []
        
        for span in all_spans:
            text = span["text"]
            
            # Skip if text is too short or contains mostly numbers
            if len(text) < 3 or re.match(r'^[\d\s\.\-]+$', text):
                continue
                
            # Check if it looks like a heading (font characteristics)
            is_bold = span["flags"] & 2**4  # Bold flag
            is_large = span["size"] >= 12
            
            # Additional heuristics for heading detection
            is_title_case = text.istitle() or text.isupper()
            is_short_line = len(text) < 100  # Headings are usually short
            
            if (is_bold and is_large) or (is_large and is_title_case and is_short_line):
                heading_candidates.append(span)
        
        # Group by font size to determine hierarchy
        size_groups = defaultdict(list)
        for candidate in heading_candidates:
            size_groups[candidate["size"]].append(candidate)
        
        # Sort font sizes in descending order
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Map font sizes to heading levels
        size_to_level = {}
        level_names = ["H1", "H2", "H3"]
        
        for i, size in enumerate(sorted_sizes[:3]):  # Only top 3 sizes
            size_to_level[size] = level_names[i]
        
        # Build outline
        outline = []
        for candidate in heading_candidates:
            level = size_to_level.get(candidate["size"])
            if level:
                outline.append({
                    "level": level,
                    "text": candidate["text"],
                    "page": candidate["page"]
                })
        
        # Sort by page number
        outline.sort(key=lambda x: x["page"])
        
        return outline
    
    def process_all_pdfs(self):
        """Process all PDFs in input directory"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in input directory")
            return
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file.name}...")
                
                # Extract outline
                outline_data = self.extract_outline(str(pdf_file))
                
                # Generate output filename
                output_filename = pdf_file.stem + ".json"
                output_path = self.output_dir / output_filename
                
                # Save JSON output
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Generated {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")

def main():
    extractor = PDFOutlineExtractor()
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()
