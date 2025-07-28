import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Tuple, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PDFOutlineExtractor:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
        self.input_dir = Path(f"{base_dir}/1b_input")
        self.output_dir = Path(f"{base_dir}/output")

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file {self.config_path} not found!")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            raise

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts but keep more characters for content
        text = re.sub(r'[^\w\s\-\.\(\)\[\]\{\}:;,\u2013\u2014\u2019\u201c\u201d\!\?\/\@\#\$\%\^\&\*\+\=\<\>\~\`]', '', text)
        
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

    def extract_text_between_headings(self, doc, start_page, end_page, start_bbox=None, end_bbox=None):
        """Extract text content between two headings"""
        content_text = []
        
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num - 1]  # fitz uses 0-based indexing
            page_text = page.get_text()
            
            if page_num == start_page and start_bbox:
                # For the starting page, get text after the heading
                blocks = page.get_text("dict")["blocks"]
                page_content = []
                start_y = start_bbox[3]  # bottom of heading bbox
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        line_bbox = line["bbox"]
                        if line_bbox[1] > start_y:  # Line starts below the heading
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            if line_text.strip():
                                page_content.append(line_text.strip())
                
                content_text.extend(page_content)
                
            elif page_num == end_page and end_bbox:
                # For the ending page, get text before the next heading
                blocks = page.get_text("dict")["blocks"]
                page_content = []
                end_y = end_bbox[1]  # top of next heading bbox
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        line_bbox = line["bbox"]
                        if line_bbox[3] < end_y:  # Line ends above the next heading
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            if line_text.strip():
                                page_content.append(line_text.strip())
                
                content_text.extend(page_content)
                
            else:
                # For middle pages, get all text
                lines = page_text.split('\n')
                content_text.extend([line.strip() for line in lines if line.strip()])
        
        # Clean and join the content
        full_content = ' '.join(content_text)
        return self.clean_text(full_content)

    def extract_outline_with_content(self, pdf_path):
        """Extract structured outline from PDF with section content"""
        doc = fitz.open(pdf_path)
        
        # First, extract the basic outline structure
        outline_data = self.extract_outline(pdf_path)
        
        # Now extract content for each section
        headings_with_content = []
        
        for i, heading in enumerate(outline_data["outline"]):
            heading_info = {
                "level": heading["level"],
                "text": heading["text"],
                "page": heading["page"],
                "content": ""
            }
            
            # Determine the range for content extraction
            start_page = heading["page"]
            
            # Find the next heading to determine end page
            if i + 1 < len(outline_data["outline"]):
                end_page = outline_data["outline"][i + 1]["page"]
            else:
                end_page = len(doc)  # Last section goes to end of document
            
            # Extract content between this heading and the next
            try:
                # Re-open doc to get bbox information
                page = doc[start_page - 1]
                blocks = page.get_text("dict")["blocks"]
                
                # Find the heading's bbox
                heading_bbox = None
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if heading["text"] in span["text"]:
                                heading_bbox = span["bbox"]
                                break
                        if heading_bbox:
                            break
                    if heading_bbox:
                        break
                
                # Find next heading's bbox if on same page
                next_heading_bbox = None
                if i + 1 < len(outline_data["outline"]) and outline_data["outline"][i + 1]["page"] == start_page:
                    next_heading_text = outline_data["outline"][i + 1]["text"]
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if next_heading_text in span["text"]:
                                    next_heading_bbox = span["bbox"]
                                    break
                            if next_heading_bbox:
                                break
                        if next_heading_bbox:
                            break
                
                # Extract content
                content = self.extract_text_between_headings(
                    doc, start_page, end_page, heading_bbox, next_heading_bbox
                )
                heading_info["content"] = content[:2000]  # Limit content length
                
            except Exception as e:
                print(f"Warning: Could not extract content for heading '{heading['text']}': {e}")
                heading_info["content"] = ""
            
            headings_with_content.append(heading_info)
        
        doc.close()
        
        return {
            "title": outline_data["title"],
            "outline": headings_with_content
        }

    def extract_outline(self, pdf_path):
        """Extract structured outline from PDF (original method)"""
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
        """Process all PDFs in input directory and return headings with content"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in input directory")
            return ""
        
        all_documents = []
        headings_summary = ""
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file.name}...")
                
                # Extract outline with content
                outline_data = self.extract_outline_with_content(str(pdf_file))
                all_documents.append(outline_data)
                
                # Create summary for ranking
                heading_texts = [h["text"] for h in outline_data["outline"]]
                joined_headings = ", ".join(heading_texts)
                headings_summary += f"\n\nTitle: {outline_data['title']}\nHeadings: {joined_headings}"
                
                # Generate output filename
                output_filename = pdf_file.stem + "_with_content.json"
                output_path = self.output_dir / output_filename
                
                # Save JSON output with content
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Generated {output_filename}")
                print(f"   Title: {outline_data['title']}")
                print(f"   Headings found: {len(outline_data['outline'])}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
        
        return headings_summary, all_documents, pdf_files


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


def select_top_headings_with_content(all_documents, job, top_n=6):
    """Select top headings with their content based on semantic similarity"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode(job, convert_to_tensor=True)
    results = []
    
    for doc in all_documents:
        title = doc["title"]
        headings = doc["outline"]
        
        if not headings:
            continue
        
        # Create embeddings for headings (including content in similarity calculation)
        heading_texts = []
        for h in headings:
            # Combine heading text with a portion of content for better similarity matching
            combined_text = h["text"]
            if h.get("content"):
                combined_text += " " + h["content"][:500]  # Use first 500 chars of content
            heading_texts.append(combined_text)
        
        heading_embs = model.encode(heading_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_emb, heading_embs)[0]
        
        # Rank headings with their full information
        heading_similarity_pairs = list(zip(headings, similarities.tolist()))
        ranked = sorted(heading_similarity_pairs, key=lambda x: -x[1])
        
        # Get top headings with all their information
        top_headings = []
        for heading_info, score in ranked[:top_n]:
            top_headings.append({
                "text": heading_info["text"],
                "level": heading_info["level"],
                "page": heading_info["page"],
                "content": heading_info.get("content", ""),
                "relevance_score": round(score, 4)
            })
        
        results.append({
            "title": title,
            "top_headings": top_headings
        })
    
    return results


def build_enhanced_qwen_prompt(top_headings_by_doc, persona, job):
    """Build enhanced prompt with content for Qwen"""
    docs_info = []
    
    for doc in top_headings_by_doc:
        # doc_info = f"\nTitle: {doc['title']}\n"
        # doc_info += "Headings:\n"
        doc_info = ""
        
        for i, heading in enumerate(doc["top_headings"], 1):
            doc_info += f"""{doc['title']}: {heading['text']}\n"""
            if heading.get("content"):
                # Truncate content for prompt
                content_preview = heading["content"][:300] + "..." if len(heading["content"]) > 300 else heading["content"]
                # doc_info += f"   Content: {content_preview}\n"
        
        docs_info.append(doc_info)
    
    all_docs_text = "\n".join(docs_info)
    
    prompt = (
        f"You are a {persona}. Your task is '{job}'.\n\n"
        "Based on the following document titles, headings, and their content, "
        "provide a structured analysis that helps accomplish the task. "
        "Focus on the most relevant headings and their title.\n\n"
        "Please provide:\n"
        "The best 10 headings and their titles in the folloing format:\n"
        "<Title 1>: <Heading 1>, <Title 2>: <Heading 2> ... and so on\n"
        "Documents Title and the underlying headings are(in same format are):\n"
        f"{all_docs_text}\n\n"
    )
    
    return prompt


# def main():
#     extractor = PDFOutlineExtractor()
#     headings_summary, all_documents = extractor.process_all_pdfs()
    
#     # Initialize model
#     model_id = "Qwen/Qwen2-0.5B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map="cpu",
#         torch_dtype=torch.float32
#     )

#     tokenizer.use_default_system_prompt = False

#     persona = "Travel Planner"
#     job_to_be_done = "Plan a 4 day trip for 10 college students"

#     # Select top headings with content
#     top_headings_with_content = select_top_headings_with_content(
#         all_documents, job=job_to_be_done, top_n=6
#     )
    
#     # Build enhanced prompt
#     prompt = build_enhanced_qwen_prompt(top_headings_with_content, persona, job_to_be_done)
#     print("\nEnhanced Prompt to Qwen:\n", prompt)

#     # Generate response
#     messages = [{"role": "user", "content": prompt}]
#     input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=512,  # Increased for more detailed response
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )
    
#     response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
#     print("\n" + "="*60)
#     print("QWEN ANALYSIS RESULT:")
#     print("="*60)
#     print(response)

#     # Save final analysis
#     final_output = {
#         "persona": persona,
#         "job_to_be_done": job_to_be_done,
#         "selected_sections": top_headings_with_content,
#         "ai_analysis": response
#     }
    
#     final_output_path = extractor.output_dir / "final_analysis.json"
#     with open(final_output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_output, f, indent=2, ensure_ascii=False)
    
#     print(f"\n✅ Final analysis saved to: {final_output_path}")


# if __name__ == "__main__":
#     main()

def get_all_sections_ranked(all_documents, job):
    """Get all sections from all documents ranked by relevance score"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode(job, convert_to_tensor=True)
    
    all_sections = []
    
    for doc in all_documents:
        title = doc["title"]
        headings = doc["outline"]
        
        if not headings:
            continue
        
        # Create embeddings for headings (including content in similarity calculation)
        heading_texts = []
        for h in headings:
            # Combine heading text with a portion of content for better similarity matching
            combined_text = h["text"]
            if h.get("content"):
                combined_text += " " + h["content"][:500]  # Use first 500 chars of content
            heading_texts.append(combined_text)
        
        heading_embs = model.encode(heading_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_emb, heading_embs)[0]
        
        # Add each section with its relevance score
        for heading_info, score in zip(headings, similarities.tolist()):
            section = {
                "title": title,
                "heading_text": heading_info["text"],
                "level": heading_info["level"],
                "page": heading_info["page"],
                "content": heading_info.get("content", ""),
                "relevance_score": round(score, 4)
            }
            all_sections.append(section)
    
    # Sort all sections by relevance score (descending)
    all_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return all_sections

def main():
    # extractor = PDFOutlineExtractor()
    config_path = Path(f"{base_dir}/src/config.json")
    extractor = PDFOutlineExtractor(config_path)
    
    persona = extractor.config.get("persona", {}).get("role", "Analyst")
    job_to_be_done = extractor.config.get("job_to_be_done", {}).get("task", "Analyze documents")
    
    print(f"\n🎯 Persona: {persona}")
    print(f"🎯 Job to be done: {job_to_be_done}")
    
    headings_summary, all_documents, pdf_files = extractor.process_all_pdfs()

    persona = "Travel Planner"
    job_to_be_done = "Plan a 4 day trip for 10 college students"

    # Get all headings with content from all documents
    all_sections_with_scores = get_all_sections_ranked(all_documents, job_to_be_done)
    
    # Select top 10 sections across all documents
    top_10_sections = all_sections_with_scores[:10]
    
    print("\n" + "="*80)
    print("TOP 10 MOST RELEVANT SECTIONS:")
    print("="*80)
    
    for i, section in enumerate(top_10_sections, 1):
        print(f"\n{i}. {section['title']} - {section['heading_text']}")
        print(f"   Page: {section['page']} | Relevance Score: {section['relevance_score']}")
        print(f"   Content Preview: {section['content'][:200]}...")
        print("-" * 60)

    # Create a mapping from document title to PDF filename for reference
    title_to_filename = {}
    for i, doc in enumerate(all_documents):
        title_to_filename[doc['title']] = pdf_files[i].name if i < len(pdf_files) else f"document_{i+1}.pdf"

    # Save final analysis with only top 10 sections
    final_output = {
        "metadata": {
            "input_documents": [str(pdf_file) for pdf_file in pdf_files],  # Convert Path objects to strings
            "persona": persona,
            "job_to_be_done": job_to_be_done
        },
        "extracted_sections": [
            {
                "document": title_to_filename.get(section["title"], section["title"]),
                "section_title": section["heading_text"],
                "importance_rank": i + 1,
                "page_number": section["page"],
            }
            for i, section in enumerate(top_10_sections)
        ],
        "subsection_analysis": [
            {
                "document": title_to_filename.get(section["title"], section["title"]),
                "refined_text": section["content"],
                "page_number": section["page"],
            }
            for section in top_10_sections
        ]
    }
    
    final_output_path = extractor.output_dir / "final_output.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Top 10 sections analysis saved to: {final_output_path}")
    print(f"📊 Processed {len(all_documents)} documents with {len(all_sections_with_scores)} total sections")

if __name__ == "__main__":
    main()