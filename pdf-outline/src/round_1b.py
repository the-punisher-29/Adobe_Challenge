import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from collections import defaultdict, Counter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PDFOutlineExtractor:
    def __init__(self):
        self.input_dir = Path(f"{base_dir}/1b_input")
        self.output_dir = Path(f"{base_dir}/output")

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'[^\w\s\-\.\(\)\[\]\{\}:;,\u2013\u2014\u2019\u201c\u201d]', '', text)
        
        return text

    def is_likely_heading(self, span, avg_font_size, common_fonts):
        """Determine if a span is likely a heading"""
        text = span["text"].strip()
        
        # Skip very short or very long text
        if len(text) < 3 or len(text) > 200:
            return False
            
        # Skip if mostly numbers/symbols
        if re.match(r'^[\d\s\.\-\(\)]+$', text):
            return False
            
        # Skip if it's a common footer/header pattern
        if re.match(r'^(page|copyright|version|\d+\s*of\s*\d+)', text.lower()):
            return False
            
        # Check font characteristics
        is_bold = span["flags"] & 16  # Bold flag (2^4)
        is_larger = span["size"] > avg_font_size * 1.1
        is_much_larger = span["size"] > avg_font_size * 1.3
        
        # Check text patterns that suggest headings
        has_numbering = re.match(r'^\d+\.?\s+', text) or re.match(r'^\d+\.\d+\.?\s+', text)
        is_title_case = text.istitle()
        is_all_caps = text.isupper() and len(text) > 3
        
        # Check if font is different from common body text fonts
        is_different_font = span["font"] not in common_fonts[:2]  # Top 2 most common fonts
        
        # Scoring system
        score = 0
        
        if is_bold:
            score += 2
        if is_larger:
            score += 1
        if is_much_larger:
            score += 2
        if has_numbering:
            score += 2
        if is_title_case:
            score += 1
        if is_all_caps:
            score += 1
        if is_different_font:
            score += 1
            
        # Additional patterns for common heading formats
        if re.match(r'^(chapter|section|part|appendix)\s+\d+', text.lower()):
            score += 3
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$', text):  # Title case words
            score += 1
            
        return score >= 3

    def extract_outline(self, pdf_path):
        """Extract structured outline from PDF"""
        doc = fitz.open(pdf_path)
        
        # Collect all text spans with font information
        all_spans = []
        page_texts = {}
        font_sizes = []
        fonts = []
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            page_spans = []
            
            for block in blocks:
                if "lines" not in block:  # Skip image blocks
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = self.clean_text(span["text"])
                        if len(text) > 1:
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
                            font_sizes.append(span["size"])
                            fonts.append(span["font"])
            
            page_texts[page_num] = page_spans
        
        doc.close()
        
        # Calculate average font size and common fonts
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        font_counter = Counter(fonts)
        common_fonts = [font for font, count in font_counter.most_common(5)]
        
        # Extract title
        title = self._extract_title(page_texts.get(1, []), avg_font_size)
        
        # Extract headings
        outline = self._extract_headings(all_spans, avg_font_size, common_fonts)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_title(self, first_page_spans, avg_font_size):
        """Extract document title from first page"""
        if not first_page_spans:
            return "Untitled Document"
        
        # Look for title in the top portion of the first page
        top_spans = [span for span in first_page_spans if span["bbox"][1] < 200]
        
        if not top_spans:
            top_spans = first_page_spans[:10]  # Fallback to first 10 spans
        
        # Find potential title candidates
        title_candidates = []
        
        for span in top_spans:
            text = span["text"].strip()
            
            # Skip very short text or common headers
            if len(text) < 5:
                continue
                
            if re.match(r'^(copyright|version|page|©)', text.lower()):
                continue
            
            # Look for large text or bold text
            is_bold = span["flags"] & 16
            is_large = span["size"] > avg_font_size * 1.2
            
            if is_large or is_bold:
                title_candidates.append({
                    "text": text,
                    "size": span["size"],
                    "bold": is_bold,
                    "y_pos": span["bbox"][1]
                })
        
        if not title_candidates:
            # Fallback: use the largest text on first page
            max_size = max(span["size"] for span in first_page_spans)
            title_candidates = [
                {"text": span["text"], "size": span["size"], "bold": False, "y_pos": span["bbox"][1]}
                for span in first_page_spans 
                if span["size"] == max_size and len(span["text"]) > 3
            ]
        
        if title_candidates:
            # Sort by size (descending) and then by position (ascending)
            title_candidates.sort(key=lambda x: (-x["size"], x["y_pos"]))
            
            # Combine multiple title parts if they're close to each other
            main_title = title_candidates[0]["text"]
            
            # Look for additional title parts
            for candidate in title_candidates[1:3]:
                if (candidate["size"] >= title_candidates[0]["size"] * 0.9 and
                    abs(candidate["y_pos"] - title_candidates[0]["y_pos"]) < 50):
                    main_title += " " + candidate["text"]
            
            return self.clean_text(main_title)
        
        return "Untitled Document"
    
    def _extract_headings(self, all_spans, avg_font_size, common_fonts):
        """Extract hierarchical headings from all spans"""
        # Filter potential headings
        heading_candidates = []
        
        for span in all_spans:
            if self.is_likely_heading(span, avg_font_size, common_fonts):
                heading_candidates.append(span)
        
        if not heading_candidates:
            return []
        
        # Remove duplicates (same text on same page)
        unique_headings = []
        seen = set()
        
        for heading in heading_candidates:
            key = (heading["text"], heading["page"])
            if key not in seen:
                seen.add(key)
                unique_headings.append(heading)
        
        # Sort by page and then by y-position
        unique_headings.sort(key=lambda x: (x["page"], x["bbox"][1]))
        
        # Determine hierarchy based on font size and text patterns
        outline = []
        
        # Group headings by font size
        size_groups = defaultdict(list)
        for heading in unique_headings:
            size_groups[heading["size"]].append(heading)
        
        # Sort sizes in descending order
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Create size to level mapping
        size_to_level = {}
        level_names = ["H1", "H2", "H3", "H4", "H5"]
        
        for i, size in enumerate(sorted_sizes[:len(level_names)]):
            size_to_level[size] = level_names[i]
        
        # Process headings and assign levels
        for heading in unique_headings:
            text = heading["text"]
            
            # Determine level based on font size
            level = size_to_level.get(heading["size"], "H3")
            
            # Override level based on text patterns
            if re.match(r'^\d+\.\s+', text):  # "1. Introduction"
                level = "H1"
            elif re.match(r'^\d+\.\d+\s+', text):  # "1.1 Overview"
                level = "H2"
            elif re.match(r'^\d+\.\d+\.\d+\s+', text):  # "1.1.1 Details"
                level = "H3"
            
            outline.append({
                "level": level,
                "text": text,
                "page": heading["page"]
            })
        
        return outline
    
    def process_all_pdfs(self):
        """Process all PDFs in input directory"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in input directory")
            return ""
        
        headings = ""
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file.name}...")
                
                # Extract outline
                outline_data = self.extract_outline(str(pdf_file))

                heading_texts = [h["text"] for h in outline_data["outline"]]
                joined_headings = ", ".join(heading_texts)

                headings += f"\n\nTitle: {outline_data['title']}\nHeadings: {joined_headings}"

                # Generate output filename
                output_filename = pdf_file.stem + ".json"
                output_path = self.output_dir / output_filename
                
                # Save JSON output
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Generated {output_filename}")
                print(f"   Title: {outline_data['title']}")
                print(f"   Headings found: {len(outline_data['outline'])}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")

        return headings

def get_user_inputs() -> Tuple[str, str]:
    """Get persona and job-to-be-done from user input"""
    print("\n" + "="*60)
    print("INTELLIGENT DOCUMENT ANALYST")
    print("="*60)

    print("\nPlease provide the following information:")

    # Get persona
    print("\n1. PERSONA DEFINITION:")
    print("   (Describe the role, expertise, and focus areas)")
    print("   Example: 'PhD Researcher in Computational Biology'")
    persona = input("   Enter persona: ").strip()

    if not persona:
        persona = "Research Analyst"
        print(f"   Using default persona: {persona}")

    # Get job-to-be-done
    print("\n2. JOB-TO-BE-DONE:")
    print("   (Describe the specific task or objective)")
    print("   Example: 'Prepare a comprehensive literature review focusing on methodologies'")
    job_to_be_done = input("   Enter job-to-be-done: ").strip()

    if not job_to_be_done:
        job_to_be_done = "Analyze and summarize key information from the documents"
        print(f"   Using default job: {job_to_be_done}")

    return persona, job_to_be_done

# def main():
#     extractor = PDFOutlineExtractor()
#     headings = extractor.process_all_pdfs()
#     model_id = "Qwen/Qwen2-0.5B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="cpu",
#         torch_dtype=torch.float32
#     )

#     tokenizer.use_default_system_prompt = False

#     # persona, job_to_be_done = get_user_inputs()
#     persona = "Travel Planner"

#     job_to_be_done = "Plan a 4 day trip for 10 college students"

#     content = f"""Your persona is of a {persona}. The job that you have to do is find the relevance of the given headings to {job_to_be_done}.
#         And then return in decending order in a json format.
#         The headings along with title of different pdfs is:
#         {headings}
#     """

#     print(content)

#     messages = [
#         {"role": "user", "content": content}
#     ]
    
#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt"
#     ).to(model.device)

#     # Generate response
#     with torch.no_grad():
#         output = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )

#     # Decode and print
#     response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
#     print("Qwen2:", response)

from sentence_transformers import SentenceTransformer, util

def rank_headings_with_embeddings(headings_str: str, query: str = "Plan a 4 day trip for 10 college students"):
    """Rank extracted headings using semantic similarity to the task query"""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Parse the headings_str into individual docs
    documents = []
    for doc_block in headings_str.strip().split("\n\n"):
        if "Title:" not in doc_block or "Headings:" not in doc_block:
            continue
        title_line, heading_line = doc_block.strip().split("\n")
        title = title_line.replace("Title:", "").strip()
        headings = [h.strip() for h in heading_line.replace("Headings:", "").split(",") if h.strip()]
        documents.append((title, headings))

    query_emb = model.encode(query, convert_to_tensor=True)
    results = []

    for title, headings in documents:
        heading_embs = model.encode(headings, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_emb, heading_embs)[0]  # shape: [N]
        
        ranked = sorted(zip(headings, similarities.tolist()), key=lambda x: -x[1])
        results.append({
            "title": title,
            "ranked_headings": [{"heading": h, "score": round(s, 4)} for h, s in ranked]
        })

    return results

def main():
    extractor = PDFOutlineExtractor()
    headings = extractor.process_all_pdfs()

    persona = "Travel Planner"
    job_to_be_done = "Plan a 4 day trip for 10 college students"

    print("\n📘 Ranking headings using sentence embeddings...\n")

    ranked = rank_headings_with_embeddings(headings, query=job_to_be_done)

    for doc in ranked:
        print(f"\n📄 Title: {doc['title']}")
        for item in doc["ranked_headings"][:5]:  # Top 5 headings
            print(f"   - {item['heading']} (score: {item['score']})")

if __name__ == "__main__":
    main()