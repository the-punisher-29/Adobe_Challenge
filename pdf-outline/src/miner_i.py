import time
import glob
import PyPDF2
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract 
import os
import json
import re
from collections import Counter, defaultdict
import statistics

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PDFStructureExtractor:
    def __init__(self):
        # Common words to filter out (headers, footers, logos, copyright)
        self.filter_words = {
            'copyright', '©', 'all rights reserved', 'page', 'confidential',
            'draft', 'proprietary', 'internal use only', 'www', 'http', 'https',
            'email', 'phone', 'tel', 'fax', 'inc', 'ltd', 'llc', 'corp',
            'company', 'organization', 'university', 'college', 'department'
        }
        
        # Common OCR artifacts from images/logos
        self.ocr_artifacts = {
            'logo', 'brand', 'trademark', '™', '®', 'established', 'since',
            'quality', 'excellence', 'innovation', 'solutions', 'services'
        }
        
        self.page_elements = []
        self.font_stats = {}
        
    def text_extraction_with_metadata(self, element):
        """Enhanced text extraction with detailed font metadata"""
        line_text = element.get_text().strip()
        
        if not line_text:
            return None
            
        font_info = []
        char_count = 0
        
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                for character in text_line:
                    if isinstance(character, LTChar):
                        font_info.append({
                            'fontname': character.fontname,
                            'size': round(character.size, 2),
                            'char': character.get_text(),
                            'bold': self.is_bold_font(character.fontname),
                            'italic': self.is_italic_font(character.fontname)
                        })
                        char_count += 1
        
        if not font_info:
            return None
            
        # Calculate dominant font characteristics
        font_sizes = [f['size'] for f in font_info]
        font_names = [f['fontname'] for f in font_info]
        bold_chars = sum(1 for f in font_info if f['bold'])
        
        dominant_size = statistics.mode(font_sizes) if font_sizes else 0
        dominant_font = statistics.mode(font_names) if font_names else ''
        bold_ratio = bold_chars / len(font_info) if font_info else 0
        
        word_count = len(line_text.split())
        
        metadata = {
            'text': line_text,
            'word_count': word_count,
            'char_count': char_count,
            'dominant_font_size': dominant_size,
            'dominant_font_name': dominant_font,
            'bold_ratio': bold_ratio,
            'is_mostly_bold': bold_ratio > 0.5,
            'bbox': element.bbox,
            'font_details': font_info
        }
        
        return metadata
    
    def is_bold_font(self, font_name):
        """Check if font name indicates bold text"""
        if not font_name:
            return False
        font_lower = font_name.lower()
        return any(keyword in font_lower for keyword in ['bold', 'heavy', 'black', 'extra'])
    
    def is_italic_font(self, font_name):
        """Check if font name indicates italic text"""
        if not font_name:
            return False
        font_lower = font_name.lower()
        return any(keyword in font_lower for keyword in ['italic', 'oblique', 'slant'])
    
    def should_filter_text(self, text):
        """Filter out common headers, footers, and OCR artifacts"""
        text_lower = text.lower()
        
        # Filter very short or very long texts
        if len(text.split()) < 1 or len(text.split()) > 20:
            return True
            
        # Filter texts with mostly numbers or special characters
        if re.match(r'^[\d\s\-\.\/\(\)]+$', text):
            return True
            
        # Filter common unwanted patterns
        if any(word in text_lower for word in self.filter_words | self.ocr_artifacts):
            return True
            
        # Filter email addresses, URLs, phone numbers
        if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text) or \
           re.search(r'https?://\S+', text) or \
           re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            return True
            
        # Filter page numbers (standalone numbers)
        if re.match(r'^\d+$', text.strip()):
            return True
            
        return False
    
    def analyze_font_statistics(self, all_elements):
        """Analyze font statistics across the document"""
        font_sizes = []
        font_names = []
        
        for element in all_elements:
            if element and not self.should_filter_text(element['text']):
                font_sizes.append(element['dominant_font_size'])
                font_names.append(element['dominant_font_name'])
        
        if not font_sizes:
            return {}
            
        self.font_stats = {
            'avg_size': statistics.mean(font_sizes),
            'median_size': statistics.median(font_sizes),
            'max_size': max(font_sizes),
            'min_size': min(font_sizes),
            'size_std': statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0,
            'common_fonts': Counter(font_names).most_common(5)
        }
        
        return self.font_stats
    
    def classify_heading_level(self, element):
        """Classify text as title or heading level based on font characteristics"""
        if not self.font_stats:
            return None
            
        size = element['dominant_font_size']
        is_bold = element['is_mostly_bold']
        word_count = element['word_count']
        
        avg_size = self.font_stats['avg_size']
        max_size = self.font_stats['max_size']
        
        # Size thresholds
        size_ratio = size / avg_size if avg_size > 0 else 1
        
        # Rule-based classification
        if word_count > 20:  # Too long to be a heading
            return None
            
        # Title detection (largest fonts, usually bold, moderate word count)
        if (size >= max_size * 0.9 and is_bold and 2 <= word_count <= 10):
            return 'title'
        
        # H1 - Large fonts, bold
        elif (size_ratio >= 1.5 and is_bold and 1 <= word_count <= 8):
            return 'H1'
        
        # H2 - Medium-large fonts, usually bold
        elif (size_ratio >= 1.2 and (is_bold or size >= avg_size * 1.3) and 1 <= word_count <= 10):
            return 'H2'
        
        # H3 - Slightly larger than average, may or may not be bold
        elif ((size_ratio >= 1.1 or is_bold) and word_count <= 12):
            return 'H3'
        
        return None
    
    def extract_table(self, pdf_path, page_num, table_num):
        """Extract table from PDF page"""
        pdf = pdfplumber.open(pdf_path)
        table_page = pdf.pages[page_num]
        tables = table_page.extract_tables()
        if table_num < len(tables):
            return tables[table_num]
        return None
    
    def table_converter(self, table):
        """Convert table to markdown format"""
        if not table:
            return ''
        table_string = ''
        for row in table:
            cleaned_row = [
                item.replace('\n', ' ') if item is not None and '\n' in item 
                else 'None' if item is None else str(item) for item in row
            ]
            table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
        return table_string.rstrip('\n')
    
    def is_element_inside_any_table(self, element, page, tables):
        """Check if element is inside a table"""
        if not tables:
            return False
        x0, y0up, x1, y1up = element.bbox
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for table in tables:
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return True
        return False
    
    def process_pdf_structure(self, pdf_path):
        """Main function to process PDF and extract structure"""
        pdfFileObj = open(pdf_path, 'rb')
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)
        
        all_elements = []
        
        # First pass: Extract all text elements with metadata
        for pagenum, page in enumerate(extract_pages(pdf_path)):
            pageObj = pdfReaded.pages[pagenum]
            
            # Handle tables
            pdf = pdfplumber.open(pdf_path)
            page_tables = pdf.pages[pagenum]
            tables = page_tables.find_tables()
            
            # Sort elements by vertical position (top to bottom)
            page_elements = [(element.y1, element) for element in page._objs]
            page_elements.sort(key=lambda a: a[0], reverse=True)
            
            for i, component in enumerate(page_elements):
                element = component[1]
                
                # Skip elements inside tables
                if self.is_element_inside_any_table(element, page, tables):
                    continue
                
                if isinstance(element, LTTextContainer):
                    metadata = self.text_extraction_with_metadata(element)
                    if metadata and not self.should_filter_text(metadata['text']):
                        metadata['page'] = pagenum + 1
                        all_elements.append(metadata)
        
        pdfFileObj.close()
        
        # Second pass: Analyze font statistics
        self.analyze_font_statistics(all_elements)
        
        # Third pass: Classify headings and extract structure
        structure = {
            'title': '',
            'outline': []
        }
        
        title_found = False
        
        for element in all_elements:
            classification = self.classify_heading_level(element)
            
            if classification == 'title' and not title_found:
                structure['title'] = element['text']
                title_found = True
            elif classification in ['H1', 'H2', 'H3']:
                structure['outline'].append({
                    'level': classification,
                    'text': element['text'],
                    'page': element['page']
                })
        
        return structure, all_elements
    
    def save_results(self, structure, metadata, output_path):
        """Save results to JSON files"""
        # Save main structure
        structure_path = output_path.replace('.txt', '_structure.json')
        with open(structure_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        # Save detailed metadata
        metadata_path = output_path.replace('.txt', '_metadata.json')
        detailed_metadata = {
            'font_statistics': self.font_stats,
            'elements': metadata,
            'total_elements': len(metadata)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_metadata, f, indent=2, ensure_ascii=False)
        
        return structure_path, metadata_path

def process_pdf_enhanced(pdf_path, output_path):
    """Enhanced PDF processing with structure extraction"""
    print(f'Processing {os.path.basename(pdf_path)} for structure extraction...')
    
    extractor = PDFStructureExtractor()
    
    try:
        structure, metadata = extractor.process_pdf_structure(pdf_path)
        structure_path, metadata_path = extractor.save_results(structure, metadata, output_path)
        
        print(f'Structure saved to: {structure_path}')
        print(f'Metadata saved to: {metadata_path}')
        print(f'Title found: {structure["title"]}')
        print(f'Headings found: {len(structure["outline"])}')
        
        return structure_path, metadata_path
        
    except Exception as e:
        print(f'Error processing {pdf_path}: {str(e)}')
        return None, None

if __name__ == '__main__':
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
    # Process all PDFs in the input directory
    for pdf_path in pdf_files:
        filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
        output_path = os.path.join(output_dir, filename)
        
        start_time = time.time()
        structure_path, metadata_path = process_pdf_enhanced(pdf_path, output_path)
        elapsed = time.time() - start_time
        
        if structure_path:
            print(f'Processing completed in {elapsed:.2f} seconds')
        print('-' * 50)