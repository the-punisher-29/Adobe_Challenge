import time
import glob
import json
import os
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextBox
import pdfplumber

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class FontInfo:
    """Detailed font information"""
    size: float
    font_family: str
    is_bold: bool
    is_italic: bool
    char_count: int
    avg_char_width: float
    line_height: float
    
@dataclass
class TextElement:
    """Text element with positioning and formatting"""
    text: str
    font_info: FontInfo
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    bbox_area: float
    is_centered: bool
    is_indented: bool
    line_spacing_above: float
    line_spacing_below: float

class PDFHeadingExtractor:
    def __init__(self):
        # Comprehensive patterns for different heading styles
        self.heading_patterns = {
            'numbered': [
                r'^(\d+\.?\s+)',                    # 1. Introduction
                r'^(\d+\.\d+\.?\s+)',              # 1.1. Subsection
                r'^(\d+\.\d+\.\d+\.?\s+)',         # 1.1.1. Sub-subsection
                r'^([IVX]+\.?\s+)',                # I. Roman numerals
                r'^([A-Z]\.?\s+)',                 # A. Letter numbering
                r'^(\([a-z]\)\s+)',                # (a) Letter in parentheses
                r'^(\d+\)\s+)',                    # 1) Number with parenthesis
            ],
            'bullet_like': [
                r'^([•▪▫■□▲►◆◇○●]\s+)',           # Various bullet points
                r'^([-–—]\s+)',                    # Dash bullets
                r'^(\*\s+)',                       # Asterisk bullets
            ],
            'chapter_section': [
                r'^(Chapter\s+\d+)',               # Chapter 1
                r'^(Section\s+\d+)',               # Section 1
                r'^(Part\s+[IVX]+)',               # Part I
                r'^(Appendix\s+[A-Z])',            # Appendix A
            ],
            'title_case': [
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Title Case Words
            ]
        }
        
        # Common non-heading patterns to exclude
        self.exclusion_patterns = [
            r'^\d+\s*$',                          # Just numbers
            r'^page\s+\d+',                       # Page numbers
            r'^figure\s+\d+',                     # Figure captions
            r'^table\s+\d+',                      # Table captions
            r'^\w+@\w+\.\w+',                     # Email addresses
            r'^https?://',                        # URLs
            r'^\d{1,2}/\d{1,2}/\d{2,4}',         # Dates
        ]
        
        # Thread-local storage for performance
        self.local = threading.local()
        
    def extract_text_with_advanced_font_info(self, element: LTTextContainer, page_bbox: Tuple[float, float, float, float]) -> Optional[TextElement]:
        """Extract text with comprehensive font and positioning analysis"""
        text = element.get_text().strip()
        if not text or len(text) < 2:
            return None
        
        font_chars = []
        char_widths = []
        
        # Collect detailed character information
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                for character in text_line:
                    if isinstance(character, LTChar):
                        font_chars.append({
                            'fontname': character.fontname,
                            'size': character.size,
                            'width': character.width,
                            'height': character.height,
                            'char': character.get_text()
                        })
                        char_widths.append(character.width)
        
        if not font_chars:
            return None
        
        # Analyze font characteristics
        sizes = [c['size'] for c in font_chars]
        fonts = [c['fontname'] for c in font_chars]
        
        dominant_size = max(set(sizes), key=sizes.count)
        dominant_font = max(set(fonts), key=fonts.count)
        
        # Determine bold/italic from font name
        font_lower = dominant_font.lower()
        is_bold = any(keyword in font_lower for keyword in ['bold', 'black', 'heavy', 'demi'])
        is_italic = any(keyword in font_lower for keyword in ['italic', 'oblique', 'slant'])
        
        # Calculate positioning metrics
        x0, y0, x1, y1 = element.bbox
        bbox_area = (x1 - x0) * (y1 - y0)
        
        # Page dimensions for relative positioning
        page_width = page_bbox[2] - page_bbox[0]
        page_height = page_bbox[3] - page_bbox[1]
        
        # Check if text is centered (within 20% of page center)
        text_center_x = (x0 + x1) / 2
        page_center_x = page_bbox[0] + page_width / 2
        is_centered = abs(text_center_x - page_center_x) < (page_width * 0.2)
        
        # Check if text is significantly indented
        left_margin = x0 - page_bbox[0]
        is_indented = left_margin > (page_width * 0.1)
        
        # Calculate average character width and line height
        avg_char_width = sum(char_widths) / len(char_widths) if char_widths else 0
        line_height = y1 - y0
        
        font_info = FontInfo(
            size=round(dominant_size, 1),
            font_family=dominant_font,
            is_bold=is_bold,
            is_italic=is_italic,
            char_count=len(text),
            avg_char_width=avg_char_width,
            line_height=line_height
        )
        
        return TextElement(
            text=text,
            font_info=font_info,
            page_num=0,  # Will be set later
            x0=x0, y0=y0, x1=x1, y1=y1,
            bbox_area=bbox_area,
            is_centered=is_centered,
            is_indented=is_indented,
            line_spacing_above=0,  # Will be calculated later
            line_spacing_below=0   # Will be calculated later
        )
    
    def calculate_line_spacing(self, elements: List[TextElement]) -> List[TextElement]:
        """Calculate line spacing between text elements"""
        if len(elements) < 2:
            return elements
        
        # Sort by y-position (top to bottom)
        sorted_elements = sorted(elements, key=lambda e: e.y1, reverse=True)
        
        for i in range(len(sorted_elements)):
            current = sorted_elements[i]
            
            # Calculate spacing above
            if i > 0:
                above = sorted_elements[i-1]
                current.line_spacing_above = above.y0 - current.y1
            
            # Calculate spacing below
            if i < len(sorted_elements) - 1:
                below = sorted_elements[i+1]
                current.line_spacing_below = current.y0 - below.y1
        
        return sorted_elements
    
    def analyze_document_structure(self, all_elements: List[TextElement]) -> Dict:
        """Comprehensive analysis of document structure and font patterns"""
        if not all_elements:
            return {}
        
        # Collect font statistics
        font_size_counter = Counter()
        font_family_counter = Counter()
        bold_sizes = Counter()
        
        # Position and spacing analysis
        line_spacings = []
        font_size_positions = defaultdict(list)
        
        for elem in all_elements:
            font_info = elem.font_info
            font_size_counter[font_info.size] += font_info.char_count
            font_family_counter[font_info.font_family] += 1
            
            if font_info.is_bold:
                bold_sizes[font_info.size] += font_info.char_count
            
            # Collect spacing information
            if elem.line_spacing_above > 0:
                line_spacings.append(elem.line_spacing_above)
            
            # Map font sizes to their typical positions
            font_size_positions[font_info.size].append({
                'page': elem.page_num,
                'y_pos': elem.y1,
                'is_centered': elem.is_centered,
                'is_bold': font_info.is_bold,
                'text_length': len(elem.text.split())
            })
        
        # Determine body text characteristics
        total_chars = sum(font_size_counter.values())
        body_text_size = None
        body_text_threshold = total_chars * 0.15  # At least 15% of content
        
        for size, count in font_size_counter.most_common():
            if count >= body_text_threshold:
                body_text_size = size
                break
        
        if body_text_size is None:
            body_text_size = font_size_counter.most_common(1)[0][0]
        
        # Analyze typical line spacing
        avg_line_spacing = sum(line_spacings) / len(line_spacings) if line_spacings else 12
        large_spacing_threshold = avg_line_spacing * 1.5
        
        # Create sophisticated font hierarchy
        heading_candidates = {}
        for size, count in font_size_counter.items():
            if size > body_text_size:
                # Calculate heading score based on multiple factors
                score = 0
                
                # Size difference from body text
                size_factor = (size - body_text_size) / body_text_size
                score += size_factor * 10
                
                # Bold usage frequency
                bold_ratio = bold_sizes.get(size, 0) / count
                score += bold_ratio * 5
                
                # Positioning patterns
                positions = font_size_positions[size]
                centered_ratio = sum(1 for p in positions if p['is_centered']) / len(positions)
                score += centered_ratio * 3
                
                # Text length patterns (headings are typically shorter)
                avg_word_count = sum(p['text_length'] for p in positions) / len(positions)
                if 2 <= avg_word_count <= 10:
                    score += 2
                
                # Frequency (headings should be less frequent than body text)
                frequency_ratio = count / total_chars
                if frequency_ratio < 0.1:  # Less than 10% of content
                    score += 2
                
                heading_candidates[size] = score
        
        # Sort and assign hierarchy levels
        sorted_candidates = sorted(heading_candidates.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        hierarchy = {}
        levels = ['H1', 'H2', 'H3']
        for i, (size, score) in enumerate(sorted_candidates[:3]):
            if score > 5:  # Minimum score threshold
                hierarchy[size] = levels[i]
        
        return {
            'body_text_size': body_text_size,
            'hierarchy': hierarchy,
            'avg_line_spacing': avg_line_spacing,
            'large_spacing_threshold': large_spacing_threshold,
            'font_families': dict(font_family_counter.most_common(5)),
            'heading_candidates': heading_candidates
        }
    
    def is_likely_heading(self, element: TextElement, doc_structure: Dict) -> Tuple[bool, Optional[str]]:
        """Advanced heading detection with multiple heuristics"""
        text = element.text.strip()
        font_info = element.font_info
        
        # Basic filters
        if len(text) < 3 or len(text) > 300:
            return False, None
        
        # Check exclusion patterns
        text_lower = text.lower()
        for pattern in self.exclusion_patterns:
            if re.match(pattern, text_lower):
                return False, None
        
        # Initialize scoring
        heading_score = 0
        confidence_factors = []
        
        # 1. Font size hierarchy check
        hierarchy = doc_structure.get('hierarchy', {})
        if font_info.size in hierarchy:
            heading_score += 15
            confidence_factors.append(f"size_hierarchy_{hierarchy[font_info.size]}")
        
        # 2. Font formatting
        if font_info.is_bold:
            heading_score += 8
            confidence_factors.append("bold")
        
        if font_info.is_italic and not font_info.is_bold:
            heading_score += 3
            confidence_factors.append("italic")
        
        # 3. Positioning analysis
        if element.is_centered:
            heading_score += 6
            confidence_factors.append("centered")
        
        if element.line_spacing_above > doc_structure.get('large_spacing_threshold', 18):
            heading_score += 5
            confidence_factors.append("large_spacing_above")
        
        if element.line_spacing_below > doc_structure.get('avg_line_spacing', 12) * 1.2:
            heading_score += 3
            confidence_factors.append("spacing_below")
        
        # 4. Text pattern analysis
        word_count = len(text.split())
        
        # Check for numbered patterns
        for category, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    if category == 'numbered':
                        heading_score += 10
                    elif category == 'chapter_section':
                        heading_score += 12
                    elif category == 'bullet_like':
                        heading_score += 4
                    confidence_factors.append(f"pattern_{category}")
                    break
        
        # 5. Text characteristics
        if text.isupper() and 3 <= word_count <= 8:
            heading_score += 7
            confidence_factors.append("all_caps")
        
        if text.istitle() and 2 <= word_count <= 12:
            heading_score += 5
            confidence_factors.append("title_case")
        
        if not text.endswith('.') and not text.endswith(','):
            heading_score += 3
            confidence_factors.append("no_ending_punct")
        
        # 6. Word count optimization
        if 1 <= word_count <= 2:
            heading_score += 3
        elif 3 <= word_count <= 8:
            heading_score += 5
        elif 9 <= word_count <= 15:
            heading_score += 2
        else:
            heading_score -= 3
        
        # 7. Font size relative to body text
        body_size = doc_structure.get('body_text_size', 12)
        if font_info.size > body_size * 1.2:
            size_ratio = font_info.size / body_size
            heading_score += min(10, int(size_ratio * 3))
            confidence_factors.append(f"large_font_{size_ratio:.1f}x")
        
        # 8. Contextual analysis - check surrounding elements
        # (This would require passing more context, simplified here)
        
        # Determine heading level
        level = None
        if font_info.size in hierarchy:
            level = hierarchy[font_info.size]
        
        # Apply minimum threshold
        is_heading = heading_score >= 12
        
        return is_heading, level
    
    def extract_title_advanced(self, first_page_elements: List[TextElement], doc_structure: Dict) -> str:
        """Advanced title extraction with multiple strategies"""
        if not first_page_elements:
            return "Untitled Document"
        
        title_candidates = []
        
        for element in first_page_elements:
            text = element.text.strip()
            font_info = element.font_info
            
            if len(text) < 3 or len(text) > 100:
                continue
            
            # Score potential titles
            title_score = 0
            
            # Font size (larger is better for titles)
            body_size = doc_structure.get('body_text_size', 12)
            if font_info.size > body_size * 1.3:
                title_score += (font_info.size / body_size) * 5
            
            # Position (higher on page is better)
            if element.y1 > 700:  # Top 20% of typical page
                title_score += 10
            elif element.y1 > 600:
                title_score += 7
            elif element.y1 > 500:
                title_score += 4
            
            # Formatting
            if font_info.is_bold:
                title_score += 8
            
            if element.is_centered:
                title_score += 6
            
            # Text characteristics
            word_count = len(text.split())
            if 2 <= word_count <= 12:
                title_score += 5
            elif word_count == 1:
                title_score += 2
            
            if text.istitle() or text.isupper():
                title_score += 4
            
            if not any(char.isdigit() for char in text[:10]):  # No numbers at start
                title_score += 3
            
            # Avoid common non-title patterns
            text_lower = text.lower()
            if any(word in text_lower for word in ['page', 'chapter', 'section', 'abstract', 'table of contents']):
                title_score -= 5
            
            title_candidates.append((text, title_score, font_info.size))
        
        if title_candidates:
            # Sort by score, then by font size
            title_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # Additional validation for top candidate
            best_candidate = title_candidates[0]
            if best_candidate[1] > 10:  # Minimum score threshold
                return best_candidate[0]
        
        # Fallback: use largest text on first page
        if first_page_elements:
            largest_element = max(first_page_elements, key=lambda x: x.font_info.size)
            if len(largest_element.text.strip()) >= 3:
                return largest_element.text.strip()
        
        return "Untitled Document"
    
    def process_pdf_parallel(self, pdf_path: str, max_pages: int = 50) -> Dict:
        """Process PDF with parallel page processing for better performance"""
        try:
            start_time = time.time()
            
            all_elements = []
            first_page_elements = []
            
            # Extract elements from all pages
            pages_processed = 0
            for page_num, page in enumerate(extract_pages(pdf_path)):
                if pages_processed >= max_pages:
                    break
                
                page_elements = []
                
                # Sort elements by y-position for proper ordering
                sorted_objs = sorted(
                    [(element.y1, element) for element in page._objs if isinstance(element, LTTextContainer)],
                    key=lambda a: a[0], 
                    reverse=True
                )
                
                for y_pos, element in sorted_objs:
                    text_elem = self.extract_text_with_advanced_font_info(element, page.bbox)
                    if text_elem:
                        text_elem.page_num = page_num + 1
                        page_elements.append(text_elem)
                        
                        if page_num == 0:
                            first_page_elements.append(text_elem)
                
                # Calculate line spacing for this page
                page_elements = self.calculate_line_spacing(page_elements)
                all_elements.extend(page_elements)
                
                pages_processed += 1
            
            extraction_time = time.time() - start_time
            print(f"  Text extraction: {extraction_time:.2f}s ({pages_processed} pages)")
            
            if not all_elements:
                return {"title": "Untitled Document", "outline": []}
            
            # Analyze document structure
            analysis_start = time.time()
            doc_structure = self.analyze_document_structure(all_elements)
            analysis_time = time.time() - analysis_start
            print(f"  Structure analysis: {analysis_time:.2f}s")
            
            # Extract title
            title_start = time.time()
            title = self.extract_title_advanced(first_page_elements, doc_structure)
            title_time = time.time() - title_start
            
            # Extract headings with deduplication
            heading_start = time.time()
            outline = []
            seen_headings = set()
            heading_counts = defaultdict(int)
            
            for element in all_elements:
                is_heading, level = self.is_likely_heading(element, doc_structure)
                
                if is_heading and level:
                    text_clean = re.sub(r'\s+', ' ', element.text.strip())
                    
                    # Improved deduplication
                    text_normalized = text_clean.lower().strip()
                    if text_normalized not in seen_headings:
                        # Additional quality filters
                        if (len(text_clean) >= 3 and 
                            heading_counts[level] < 100 and  # Limit per level
                            not text_clean.isdigit()):
                            
                            outline.append({
                                "level": level,
                                "text": text_clean,
                                "page": element.page_num
                            })
                            
                            seen_headings.add(text_normalized)
                            heading_counts[level] += 1
            
            # Sort outline by page, then by level hierarchy
            level_order = {'H1': 1, 'H2': 2, 'H3': 3}
            outline.sort(key=lambda x: (x['page'], level_order.get(x['level'], 4)))
            
            heading_time = time.time() - heading_start
            total_time = time.time() - start_time
            
            print(f"  Title extraction: {title_time:.3f}s")
            print(f"  Heading extraction: {heading_time:.2f}s")
            print(f"  Total processing: {total_time:.2f}s")
            
            return {
                "title": title,
                "outline": outline,
                "metadata": {
                    "pages_processed": pages_processed,
                    "total_elements": len(all_elements),
                    "headings_found": len(outline),
                    "processing_time": round(total_time, 2),
                    "doc_structure": {
                        "body_text_size": doc_structure.get('body_text_size'),
                        "heading_levels_detected": list(doc_structure.get('hierarchy', {}).values()),
                        "font_families": list(doc_structure.get('font_families', {}).keys())[:3]
                    }
                }
            }
            
        except Exception as e:
            print(f"Error in process_pdf_parallel: {str(e)}")
            return {"title": "Processing Error", "outline": []}

def main():
    """Main function with enhanced error handling and performance monitoring"""
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        return
    
    extractor = PDFHeadingExtractor()
    
    total_start = time.time()
    successful_processes = 0
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
        
        try:
            # Process PDF
            result = extractor.process_pdf_parallel(pdf_path, max_pages=50)
            
            # Save results
            filename = os.path.splitext(os.path.basename(pdf_path))[0] + '_outline.json'
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Print summary
            metadata = result.get('metadata', {})
            print(f"✅ SUCCESS")
            print(f"   📝 Title: {result['title'][:60]}...")
            print(f"   🔍 Headings: {len(result['outline'])} found")
            print(f"   📊 Pages: {metadata.get('pages_processed', 'N/A')}")
            print(f"   ⚡ Time: {metadata.get('processing_time', 'N/A')}s")
            print(f"   💾 Output: {output_path}")
            
            # Show heading breakdown
            if result['outline']:
                level_counts = Counter(h['level'] for h in result['outline'])
                breakdown = ', '.join([f"{level}:{count}" for level, count in sorted(level_counts.items())])
                print(f"   📋 Breakdown: {breakdown}")
            
            successful_processes += 1
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            continue
    
    total_time = time.time() - total_start
    print(f"\n🎯 SUMMARY:")
    print(f"   Processed: {successful_processes}/{len(pdf_files)} files")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg per file: {total_time/len(pdf_files):.2f}s")

if __name__ == '__main__':
    main()