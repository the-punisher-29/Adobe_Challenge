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
import pickle
import json
import re
import statistics
from collections import defaultdict, Counter

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
        self.metadata = {
            'title': '',
            'pages': {},
            'outline': [],
            'text_elements': [],
            'font_analysis': defaultdict(list),
            'structure_analysis': {}
        }
        
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
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]
    return table


def table_converter(table):
    """Converts a table (list of lists) to a markdown-like string"""
    table_string = ''
    for row_num in range(len(table)):
        row = table[row_num]
        cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
    table_string = table_string[:-1]
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
    images = convert_from_path(input_file)
    image = images[0]
    output_file = 'PDF_image.png'
    image.save(output_file, 'PNG')


def image_to_text(image_path):
    """Runs OCR on an image and returns extracted text"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


def process_pdf_with_metadata(pdf_path, output_text_path, output_metadata_path, output_json_path):
    """Enhanced PDF processing with comprehensive metadata extraction"""
    pdfFileObj = open(pdf_path, 'rb')
    pdfReaded = PyPDF2.PdfReader(pdfFileObj)
    
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
        
        pdf = pdfplumber.open(pdf_path)
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
        
        for i, component in enumerate(page_elements):
            element = component[1]
            element_metadata = {
                'page': pagenum + 1,
                'position': component[0],
                'bbox': element.bbox,
                'type': 'unknown',
                'text': '',
                'formats': [],
                'is_table': False,
                'table_index': None
            }
            
            # Handle tables
            if table_in_page != -1 and is_element_inside_any_table(element, page, tables):
                table_found = find_table_for_element(element, page, tables)
                if table_found == table_in_page and table_found is not None:
                    page_content.append(text_from_tables[table_in_page])
                    page_text.append('table')
                    line_format.append('table')
                    
                    element_metadata.update({
                        'type': 'table',
                        'text': text_from_tables[table_in_page],
                        'is_table': True,
                        'table_index': table_in_page
                    })
                    
                    table_in_page += 1
                    page_elements_detailed.append(element_metadata)
                continue
            
            # Handle text elements with enhanced metadata extraction
            if not is_element_inside_any_table(element, page, tables):
                if isinstance(element, LTTextContainer):
                    # Use enhanced text extraction
                    enhanced_metadata = extractor.text_extraction_with_metadata(element)
                    
                    if enhanced_metadata:
                        # Add page and position info
                        enhanced_metadata.update({
                            'page': pagenum + 1,
                            'position': component[0],
                            'type': 'text'
                        })
                        
                        all_text_elements.append(enhanced_metadata)
                        
                        # Keep backward compatibility
                        (line_text, format_per_line) = text_extraction(element)
                        page_text.append(line_text)
                        line_format.append(format_per_line)
                        page_content.append(line_text)
                        
                        element_metadata.update({
                            'type': 'text',
                            'text': line_text,
                            'formats': format_per_line,
                            'enhanced_metadata': enhanced_metadata
                        })
                    
                # Handle images
                elif isinstance(element, LTFigure):
                    crop_image(element, pageObj)
                    convert_to_images('cropped_image.pdf')
                    image_text = image_to_text('PDF_image.png')
                    text_from_images.append(image_text)
                    page_content.append(image_text)
                    page_text.append('image')
                    line_format.append('image')
                    image_flag = True
                    
                    element_metadata.update({
                        'type': 'image',
                        'text': image_text,
                        'ocr_text': image_text
                    })
                
                page_elements_detailed.append(element_metadata)
        
        # Store page data
        dctkey = 'Page_' + str(pagenum)
        text_per_page[dctkey] = [page_text, line_format, text_from_images, text_from_tables, page_content]
        detailed_metadata[f'page_{pagenum + 1}'] = page_elements_detailed
    
    pdfFileObj.close()
    
    # Clean up temporary files
    if image_flag:
        if os.path.exists('cropped_image.pdf'):
            os.remove('cropped_image.pdf')
        if os.path.exists('PDF_image.png'):
            os.remove('PDF_image.png')
    
    # Analyze font statistics with enhanced extractor
    extractor.analyze_font_statistics(all_text_elements)
    
    # Extract title using enhanced method
    extractor.metadata['title'] = extractor.extract_title(all_text_elements)
    
    # Extract headings and build outline with improved classification
    outline = []
    for element in all_text_elements:
        if not extractor.should_filter_text(element['text']):
            heading_level = extractor.classify_heading_level(element)
            
            if heading_level and heading_level != 'title':  # Exclude title from outline
                outline.append({
                    'level': heading_level,
                    'text': element['text'].strip(),
                    'page': element['page']
                })
    
    # Create comprehensive metadata
    extractor.metadata.update({
        'pages': detailed_metadata,
        'outline': outline,
        'font_stats': extractor.font_stats,
        'total_pages': len(detailed_metadata),
        'all_text_elements': all_text_elements
    })
    
    # Write text file
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for page in text_per_page:
            result = ''.join(text_per_page[page][4])
            f.write(f'===== {page} =====\n')
            f.write(result + '\n')
    
    # Write metadata pickle file
    with open(output_metadata_path, 'wb') as f:
        pickle.dump(extractor.metadata, f)
    
    # Create JSON output
    json_output = {
        'title': extractor.metadata['title'],
        'outline': outline
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    return extractor.metadata


# Set up directories
input_dir = os.path.join(base_dir, 'input')
output_dir = os.path.join(base_dir, 'noutput')
metadata_dir = os.path.join(f"{base_dir}/noutput", 'metadata')
json_dir = os.path.join(f"{base_dir}/noutput", 'json_output')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))

if __name__ == '__main__':
    # Process all PDFs in the input directory
    for pdf_path in pdf_files:
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        output_text_path = os.path.join(output_dir, base_filename + '.txt')
        output_metadata_path = os.path.join(metadata_dir, base_filename + '_metadata.pkl')
        output_json_path = os.path.join(json_dir, base_filename + '_structure.json')
        
        print(f'Processing {os.path.basename(pdf_path)}...')
        start_time = time.time()
        
        metadata = process_pdf_with_metadata(
            pdf_path, 
            output_text_path, 
            output_metadata_path, 
            output_json_path
        )
        
        elapsed = time.time() - start_time
        print(f'Processed {os.path.basename(pdf_path)} in {elapsed:.2f} seconds.')
        print(f'  Text output: {output_text_path}')
        print(f'  Metadata: {output_metadata_path}')
        print(f'  JSON structure: {output_json_path}')
        print(f'  Found {len(metadata["outline"])} headings')
        print(f'  Document title: {metadata["title"][:50]}...' if len(metadata["title"]) > 50 else f'  Document title: {metadata["title"]}')
        print()