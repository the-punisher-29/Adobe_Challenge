import time
import glob
import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR # Changed from pytesseract to PaddleOCR
import os
import json # Changed from pickle to json
import re
import statistics
from collections import defaultdict, Counter

# Note: This script will download PaddleOCR models on its first run.

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def make_serializable(obj):
    """Recursively converts an object to be JSON serializable."""
    if isinstance(obj, (dict, defaultdict, Counter)):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(elem) for elem in obj]
    # Add other type conversions if necessary
    return obj

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
        self.metadata = {
            'title': '',
            'pages': {},
            'outline': [],
            'text_elements': [],
            'font_analysis': defaultdict(list),
            'structure_analysis': {}
        }

        # Initialize PaddleOCR
        try:
            # You can specify other languages like 'ch', 'fr', 'german', etc.
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except Exception as e:
            print(f"Could not initialize PaddleOCR. Please ensure it's installed ('pip install paddleocr paddlepaddle'). Error: {e}")
            self.ocr = None

    def image_to_text(self, image_path):
        """Runs OCR on an image using PaddleOCR and returns extracted text"""
        if not self.ocr:
            print("OCR model not available.")
            return ""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line and len(line) > 1]
                return "\n".join(lines)
        except Exception as e:
            print(f"Error during OCR processing with PaddleOCR: {e}")
        return ""
        
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
        
        # Filter very short or very long texts that are unlikely to be headings
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
    
    def extract_title(self, all_elements):
        """Extract document title from all elements with improved logic"""
        title_candidates = []
        
        for element in all_elements:
            if element and not self.should_filter_text(element['text']):
                classification = self.classify_heading_level(element)
                if classification == 'title':
                    title_candidates.append({
                        'text': element['text'].strip(),
                        'size': element['dominant_font_size'],
                        'page': element.get('page', 1),
                        'position': element.get('position', 0)
                    })
        
        if title_candidates:
            # Sort by page number first, then by font size and position
            title_candidates.sort(key=lambda x: (x['page'], -x['size'], -x['position']))
            return title_candidates[0]['text']
        
        # Fallback: look for largest text on first page
        first_page_candidates = [e for e in all_elements if e and e.get('page', 1) == 1 
                                 and not self.should_filter_text(e['text'])]
        if first_page_candidates:
            first_page_candidates.sort(key=lambda x: x['dominant_font_size'], reverse=True)
            return first_page_candidates[0]['text'].strip()
        
        return "Untitled Document"


def text_extraction(element):
    """Basic text extraction function for backward compatibility"""
    line_text = element.get_text()
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    format_per_line = list(set(line_formats))
    return (line_text, format_per_line)


def extract_table(pdf_path, page_num, table_num):
    """Extracts a table from a specific page in the PDF"""
    with pdfplumber.open(pdf_path) as pdf:
        table_page = pdf.pages[page_num]
        tables = table_page.extract_tables()
        if tables and len(tables) > table_num:
            return tables[table_num]
    return None


def table_converter(table):
    """Converts a table (list of lists) to a markdown-like string"""
    if not table:
        return ""
    table_string = ''
    for row in table:
        # Clean each cell in the row
        cleaned_row = [str(item).replace('\n', ' ') if item is not None else '' for item in row]
        table_string += '| ' + ' | '.join(cleaned_row) + ' |\n'
    return table_string


def is_element_inside_any_table(element, page, tables):
    """Checks if a PDF element is inside any detected table"""
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


def find_table_for_element(element, page, tables):
    """Finds which table (if any) a PDF element belongs to"""
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i
    return None


def crop_image(element, pageObj):
    """Crops an image region from a PDF page and saves as a new PDF"""
    [image_left, image_top, image_right, image_bottom] = [element.x0, element.y0, element.x1, element.y1]
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)


def convert_to_images(input_file):
    """Converts the first page of a PDF to an image"""
    try:
        images = convert_from_path(input_file)
        if images:
            image = images[0]
            output_file = 'PDF_image.png'
            image.save(output_file, 'PNG')
            return output_file
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
    return None


def process_pdf_with_metadata(pdf_path, output_text_path, output_metadata_path, output_json_path):
    """Enhanced PDF processing with comprehensive metadata extraction"""
    pdfFileObj = open(pdf_path, 'rb')
    pdfReaded = PyPDF2.PdfReader(pdf_path)
    
    extractor = PDFStructureExtractor()
    text_per_page = {}
    detailed_metadata = {}
    all_text_elements = []
    image_flag = False
    
    # First pass: collect all text elements and font information
    for pagenum, page in enumerate(extract_pages(pdf_path)):
        pageObj = pdfReaded.pages[pagenum]
        page_text = []
        line_format = []
        text_from_images = []
        text_from_tables = []
        page_content = []
        page_elements_detailed = []
        table_in_page = -1
        
        with pdfplumber.open(pdf_path) as pdf:
            page_tables = pdf.pages[pagenum]
            tables = page_tables.find_tables()
        
        if len(tables) != 0:
            table_in_page = 0
            
        # Extract tables
        for table_num in range(len(tables)):
            table = extract_table(pdf_path, pagenum, table_num)
            table_string = table_converter(table)
            text_from_tables.append(table_string)
        
        # Sort elements by position (top to bottom)
        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda a: a[0], reverse=True)
        
        # Keep track of which tables have been added to avoid duplication
        added_tables_indices = set()

        for i, component in enumerate(page_elements):
            element = component[1]
            
            # Check if element is part of a table
            table_index = find_table_for_element(element, page, tables)

            # Handle Tables
            if table_index is not None and table_index not in added_tables_indices:
                table_md = text_from_tables[table_index]
                page_content.append(table_md)
                page_text.append('table')
                line_format.append('table')
                
                element_metadata = {
                    'type': 'table',
                    'text': table_md,
                    'bbox': tables[table_index].bbox,
                }
                page_elements_detailed.append(element_metadata)
                added_tables_indices.add(table_index)
                continue # Skip to next element once table is processed

            # Skip text elements that are inside a table's bounding box
            if is_element_inside_any_table(element, page, tables):
                continue
            
            # Handle Text Elements
            if isinstance(element, LTTextContainer):
                enhanced_metadata = extractor.text_extraction_with_metadata(element)
                if enhanced_metadata:
                    enhanced_metadata.update({
                        'page': pagenum + 1,
                        'position': component[0],
                        'type': 'text'
                    })
                    all_text_elements.append(enhanced_metadata)
                    
                    page_content.append(enhanced_metadata['text'])
                    page_text.append(enhanced_metadata['text'])
                    page_elements_detailed.append({
                        'type': 'text',
                        'text': enhanced_metadata['text'],
                        'bbox': element.bbox,
                        'enhanced_metadata': enhanced_metadata
                    })
            
            # Handle Images
            elif isinstance(element, LTFigure):
                crop_image(element, pageObj)
                image_path = convert_to_images('cropped_image.pdf')
                if image_path:
                    image_text = extractor.image_to_text(image_path)
                    text_from_images.append(image_text)
                    page_content.append(f"\n[IMAGE OCR TEXT]:\n{image_text}\n")
                    page_text.append('image')
                    line_format.append('image')
                    image_flag = True
                    
                    page_elements_detailed.append({
                        'type': 'image',
                        'ocr_text': image_text,
                        'bbox': element.bbox,
                    })
            
        dctkey = 'Page_' + str(pagenum)
        text_per_page[dctkey] = [page_text, line_format, text_from_images, text_from_tables, page_content]
        detailed_metadata[f'page_{pagenum + 1}'] = page_elements_detailed
    
    pdfFileObj.close()
    
    if image_flag:
        if os.path.exists('cropped_image.pdf'): os.remove('cropped_image.pdf')
        if os.path.exists('PDF_image.png'): os.remove('PDF_image.png')
    
    # --- Post-processing and Structuring ---
    extractor.analyze_font_statistics(all_text_elements)
    extractor.metadata['title'] = extractor.extract_title(all_text_elements)
    
    # Add heading classification to each text element
    for element in all_text_elements:
        heading_level = None
        if not extractor.should_filter_text(element['text']):
            heading_level = extractor.classify_heading_level(element)
        element['heading_level'] = heading_level

    # --- Create Structured JSON Output ---
    text_meta_lookup = {tuple(el['bbox']): el for el in all_text_elements}
    final_structure = []
    for page_num_str, page_elements in sorted(detailed_metadata.items()):
        page_num = int(page_num_str.replace('page_', ''))
        page_structure = {"page": page_num, "elements": []}
        
        for element in page_elements:
            final_element = {"type": element['type'], "bbox": list(element['bbox'])}

            if element['type'] == 'text':
                enhanced_meta = text_meta_lookup.get(tuple(element['bbox']))
                if enhanced_meta:
                    if enhanced_meta.get('heading_level') and enhanced_meta['heading_level'] != 'title':
                        final_element['type'] = 'heading'
                        final_element['level'] = enhanced_meta['heading_level']
                    final_element['text'] = enhanced_meta['text']
                else:
                    final_element['text'] = element.get('text', '')
            
            elif element['type'] == 'table':
                final_element['markdown'] = element.get('text', '')

            elif element['type'] == 'image':
                final_element['ocr_text'] = element.get('ocr_text', '')
            
            page_structure['elements'].append(final_element)
        final_structure.append(page_structure)

    json_output = {
        'title': extractor.metadata['title'],
        'total_pages': len(detailed_metadata),
        'structure': final_structure
    }
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)

    # --- Create Full Metadata and Text File ---
    full_metadata = {
        'title': extractor.metadata['title'],
        'total_pages': len(detailed_metadata),
        'font_stats': extractor.font_stats,
        'all_text_elements': all_text_elements
    }
    
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(full_metadata), f, indent=4)

    with open(output_text_path, 'w', encoding='utf-8') as f:
        for page, data in sorted(text_per_page.items()):
            result = '\n'.join(data[4])
            f.write(f'===== {page} =====\n')
            f.write(result + '\n\n')
    
    return json_output


if __name__ == '__main__':
    # Set up single output directory
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'n1output') # Unified output directory
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))

    for pdf_path in pdf_files:
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Define paths for the three output files in the 'output' folder
        output_text_path = os.path.join(output_dir, base_filename + '.txt')
        output_metadata_path = os.path.join(output_dir, base_filename + '_full_metadata.json')
        output_json_path = os.path.join(output_dir, base_filename + '_structure.json')
        
        print(f'Processing {os.path.basename(pdf_path)}...')
        start_time = time.time()
        
        try:
            structured_data = process_pdf_with_metadata(
                pdf_path, 
                output_text_path, 
                output_metadata_path, 
                output_json_path
            )
            
            elapsed = time.time() - start_time
            print(f'Successfully processed {os.path.basename(pdf_path)} in {elapsed:.2f} seconds.')
            print(f'  Text output: {output_text_path}')
            print(f'  Full metadata: {output_metadata_path}')
            print(f'  Structured JSON: {output_json_path}')
            
            # Count headings from the final structured data
            heading_count = sum(1 for page in structured_data.get('structure', []) 
                                for element in page.get('elements', []) 
                                if element.get('type') == 'heading')
            print(f'  Found {heading_count} headings.')
            doc_title = structured_data.get('title', 'N/A')
            print(f'  Document title: {doc_title[:70]}' + ('...' if len(doc_title) > 70 else ''))
            print("-" * 50)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"!!! An error occurred while processing {os.path.basename(pdf_path)} after {elapsed:.2f} seconds.")
            print(f"Error: {e}")
            print("-" * 50)
