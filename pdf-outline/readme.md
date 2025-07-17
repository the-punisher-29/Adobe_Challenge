# PDF Outline Extractor: Analysis and Improvements

## Current Approach Analysis

### Strengths
- **Rule-based heading detection**: Uses font size, bold flags, and text patterns to identify headings
- **Hierarchical structure**: Creates proper heading levels (H1-H5) based on font characteristics
- **Text cleaning**: Removes artifacts and normalizes text
- **Title extraction**: Identifies document titles from the first page
- **Duplicate removal**: Eliminates repeated headings on the same page

### Current Algorithm Flow
1. **Text Extraction**: Uses PyMuPDF to extract text spans with font metadata
2. **Font Analysis**: Calculates average font size and identifies common fonts
3. **Heading Detection**: Scores text spans based on:
   - Font size (larger = higher score)
   - Bold formatting (flag 16)
   - Text patterns (numbering, title case, ALL CAPS)
   - Font differences from body text
4. **Hierarchy Assignment**: Groups by font size and assigns heading levels
5. **Output Generation**: Creates structured JSON with title and outline

### Limitations
- **Language-specific**: Only works well with English text patterns
- **Rule-based rigidity**: May miss context-dependent headings
- **No semantic understanding**: Cannot distinguish between actual headings and formatted text
- **Limited font heuristics**: Relies heavily on font metadata which may be inconsistent

## Proposed Improvements

### 1. Multi-Language Support

#### Language Detection
```python
from langdetect import detect
import polyglot
from polyglot.detect import Detector

class MultiLanguageSupport:
    def __init__(self):
        self.language_patterns = {
            'en': {
                'chapter': r'chapter\s+\d+',
                'section': r'section\s+\d+',
                'part': r'part\s+\d+',
                'appendix': r'appendix\s+[a-z]?'
            },
            'fr': {
                'chapter': r'chapitre\s+\d+',
                'section': r'section\s+\d+',
                'part': r'partie\s+\d+',
                'appendix': r'annexe\s+[a-z]?'
            },
            'es': {
                'chapter': r'capítulo\s+\d+',
                'section': r'sección\s+\d+',
                'part': r'parte\s+\d+',
                'appendix': r'apéndice\s+[a-z]?'
            },
            'de': {
                'chapter': r'kapitel\s+\d+',
                'section': r'abschnitt\s+\d+',
                'part': r'teil\s+\d+',
                'appendix': r'anhang\s+[a-z]?'
            }
        }
    
    def detect_language(self, text_samples):
        """Detect primary language from text samples"""
        combined_text = ' '.join(text_samples[:100])  # Use first 100 samples
        try:
            return detect(combined_text)
        except:
            return 'en'  # Default to English
    
    def get_language_patterns(self, lang_code):
        """Get language-specific patterns"""
        return self.language_patterns.get(lang_code, self.language_patterns['en'])
```

#### Unicode and Character Handling
```python
import unicodedata

def normalize_text(text, language='en'):
    """Normalize text for different languages"""
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Language-specific normalization
    if language in ['ar', 'he']:  # Right-to-left languages
        text = text[::-1]  # Reverse for processing
    
    # Remove language-specific artifacts
    if language == 'zh':  # Chinese
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
    elif language == 'ja':  # Japanese
        text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w\s]', '', text)
    elif language == 'ko':  # Korean
        text = re.sub(r'[^\uac00-\ud7af\w\s]', '', text)
    
    return text
```

### 2. Small Language Model Integration

#### Model Selection (Under 200MB)
```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SmallModelClassifier:
    def __init__(self):
        # Use DistilBERT or similar small model (~70MB)
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Pre-computed heading embeddings
        self.heading_examples = [
            "Introduction", "Chapter 1", "Overview", "Background",
            "Methodology", "Results", "Discussion", "Conclusion",
            "References", "Appendix", "Summary", "Abstract"
        ]
        
        self.heading_embeddings = self._compute_heading_embeddings()
    
    def _compute_heading_embeddings(self):
        """Pre-compute embeddings for typical headings"""
        embeddings = []
        for example in self.heading_examples:
            embedding = self._get_embedding(example)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _get_embedding(self, text):
        """Get embedding for text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def is_heading_semantic(self, text, threshold=0.7):
        """Determine if text is semantically similar to headings"""
        text_embedding = self._get_embedding(text)
        similarities = cosine_similarity(text_embedding, self.heading_embeddings)
        max_similarity = similarities.max()
        return max_similarity > threshold, max_similarity
```

#### Lightweight Document Classification
```python
class LightweightDocumentClassifier:
    def __init__(self):
        # Use a small BERT variant or custom trained model
        self.classifier = self._load_model()
        
    def _load_model(self):
        """Load pre-trained heading classifier"""
        # Option 1: Use a small pre-trained model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 23MB
        return model
        
    def classify_heading_level(self, text, font_size, is_bold, context):
        """Classify heading level using both features and semantics"""
        # Extract features
        features = {
            'font_size': font_size,
            'is_bold': is_bold,
            'length': len(text),
            'has_numbers': bool(re.search(r'\d', text)),
            'title_case': text.istitle(),
            'all_caps': text.isupper(),
            'context_position': context.get('position', 0)
        }
        
        # Get semantic embedding
        embedding = self.classifier.encode([text])
        
        # Combine rule-based and semantic scoring
        rule_score = self._calculate_rule_score(features)
        semantic_score = self._calculate_semantic_score(embedding, text)
        
        combined_score = 0.6 * rule_score + 0.4 * semantic_score
        
        return self._score_to_level(combined_score)
    
    def _calculate_rule_score(self, features):
        """Calculate rule-based score"""
        score = 0
        if features['is_bold']:
            score += 0.3
        if features['font_size'] > 12:
            score += 0.2
        if features['has_numbers']:
            score += 0.2
        if features['title_case']:
            score += 0.1
        if features['all_caps']:
            score += 0.1
        if features['length'] < 50:
            score += 0.1
        return min(score, 1.0)
    
    def _calculate_semantic_score(self, embedding, text):
        """Calculate semantic similarity score"""
        # Compare with known heading patterns
        heading_patterns = [
            "chapter introduction section",
            "conclusion summary results",
            "methodology approach analysis",
            "background literature review"
        ]
        
        pattern_embeddings = self.classifier.encode(heading_patterns)
        similarities = cosine_similarity(embedding, pattern_embeddings)
        return similarities.max()
    
    def _score_to_level(self, score):
        """Convert score to heading level"""
        if score > 0.8:
            return "H1"
        elif score > 0.6:
            return "H2"
        elif score > 0.4:
            return "H3"
        elif score > 0.2:
            return "H4"
        else:
            return "H5"
```

### 3. Performance Optimization

#### Efficient Processing Pipeline
```python
import concurrent.futures
from functools import lru_cache
import time

class OptimizedPDFProcessor:
    def __init__(self):
        self.ml_classifier = LightweightDocumentClassifier()
        self.lang_support = MultiLanguageSupport()
        
    @lru_cache(maxsize=1000)
    def cached_classification(self, text, font_size, is_bold):
        """Cache classification results"""
        return self.ml_classifier.classify_heading_level(
            text, font_size, is_bold, {}
        )
    
    def process_page_batch(self, page_batch):
        """Process multiple pages in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for page_num, page in page_batch:
                future = executor.submit(self._process_single_page, page_num, page)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
            return results
    
    def _process_single_page(self, page_num, page):
        """Process a single page efficiently"""
        blocks = page.get_text("dict")["blocks"]
        page_spans = []
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    text = self.clean_text(span["text"])
                    if len(text) > 2:
                        span_info = {
                            "text": text,
                            "size": round(span["size"], 1),
                            "font": span["font"],
                            "flags": span["flags"],
                            "page": page_num,
                            "bbox": span["bbox"]
                        }
                        page_spans.append(span_info)
        
        return page_spans
    
    def extract_outline_optimized(self, pdf_path):
        """Optimized outline extraction with ML support"""
        start_time = time.time()
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Process pages in batches
        batch_size = 10
        all_spans = []
        
        for i in range(0, total_pages, batch_size):
            batch = [(j + 1, doc[j]) for j in range(i, min(i + batch_size, total_pages))]
            batch_results = self.process_page_batch(batch)
            
            for page_spans in batch_results:
                all_spans.extend(page_spans)
        
        doc.close()
        
        # Language detection
        sample_texts = [span["text"] for span in all_spans[:50]]
        detected_lang = self.lang_support.detect_language(sample_texts)
        
        # Extract headings with ML classification
        outline = self._extract_headings_ml(all_spans, detected_lang)
        
        # Extract title
        title = self._extract_title_ml(all_spans[:50])
        
        processing_time = time.time() - start_time
        
        return {
            "title": title,
            "outline": outline,
            "language": detected_lang,
            "processing_time": processing_time,
            "total_pages": total_pages
        }
```

### 4. Enhanced Feature Engineering

#### Advanced Text Analysis
```python
class AdvancedTextAnalyzer:
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
    def extract_advanced_features(self, span, page_context):
        """Extract advanced features for ML classification"""
        text = span["text"]
        
        features = {
            # Font features
            'font_size': span["size"],
            'is_bold': bool(span["flags"] & 16),
            'is_italic': bool(span["flags"] & 4),
            'font_name': span["font"],
            
            # Text features
            'length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'has_numbers': bool(re.search(r'\d', text)),
            'starts_with_number': bool(re.match(r'^\d', text)),
            'ends_with_colon': text.endswith(':'),
            'all_caps': text.isupper(),
            'title_case': text.istitle(),
            'sentence_case': text[0].isupper() and not text.isupper(),
            
            # Position features
            'page_number': span["page"],
            'y_position': span["bbox"][1],
            'x_position': span["bbox"][0],
            'relative_y': span["bbox"][1] / page_context.get('height', 800),
            'relative_x': span["bbox"][0] / page_context.get('width', 600),
            
            # Content features
            'has_common_heading_words': self._has_heading_keywords(text),
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / max(len(text), 1),
            'stop_word_ratio': len([w for w in text.lower().split() if w in self.stop_words]) / max(len(text.split()), 1),
            
            # Structural features
            'indentation': span["bbox"][0],
            'line_spacing': page_context.get('line_spacing', 0),
            'follows_heading': page_context.get('follows_heading', False),
            'precedes_body': page_context.get('precedes_body', False)
        }
        
        return features
    
    def _has_heading_keywords(self, text):
        """Check if text contains common heading keywords"""
        heading_keywords = {
            'en': ['chapter', 'section', 'introduction', 'conclusion', 'summary', 'overview', 'background', 'methodology', 'results', 'discussion', 'references', 'appendix'],
            'fr': ['chapitre', 'section', 'introduction', 'conclusion', 'résumé', 'aperçu', 'contexte', 'méthodologie', 'résultats', 'discussion', 'références', 'annexe'],
            'es': ['capítulo', 'sección', 'introducción', 'conclusión', 'resumen', 'visión', 'antecedentes', 'metodología', 'resultados', 'discusión', 'referencias', 'apéndice'],
            'de': ['kapitel', 'abschnitt', 'einführung', 'schluss', 'zusammenfassung', 'überblick', 'hintergrund', 'methodik', 'ergebnisse', 'diskussion', 'referenzen', 'anhang']
        }
        
        for lang_keywords in heading_keywords.values():
            if any(keyword in text.lower() for keyword in lang_keywords):
                return True
        return False
```

### 5. Implementation Guidelines

#### Model Size Management
```python
# Use quantized models to reduce size
import torch.quantization

def load_quantized_model(model_name):
    """Load quantized model for smaller size"""
    model = AutoModel.from_pretrained(model_name)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

# Alternative: Use ONNX for inference
import onnxruntime as ort

def create_onnx_session(model_path):
    """Create ONNX runtime session for faster inference"""
    session = ort.InferenceSession(model_path)
    return session
```

#### Memory Management
```python
import gc
import torch

class MemoryOptimizedProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def process_with_memory_management(self, spans):
        """Process spans with memory optimization"""
        batch_size = 32
        results = []
        
        for i in range(0, len(spans), batch_size):
            batch = spans[i:i+batch_size]
            
            # Process batch
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Clear memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        return results
```

## Performance Targets

### Benchmarks for 50-page PDF (30-second target)
- **Text extraction**: 5-8 seconds
- **Language detection**: 1-2 seconds  
- **ML classification**: 15-20 seconds
- **Post-processing**: 2-3 seconds
- **Total**: 23-33 seconds

### Optimization Strategies
1. **Parallel processing**: Process pages in batches
2. **Caching**: Cache ML predictions for similar texts
3. **Early stopping**: Skip obvious non-headings
4. **Quantized models**: Use INT8 quantization
5. **ONNX runtime**: Faster inference than PyTorch

## Installation Requirements

```bash
pip install torch transformers sentence-transformers
pip install fitz PyMuPDF langdetect polyglot
pip install scikit-learn numpy
pip install onnxruntime  # Optional for ONNX optimization
```

## Usage Example

```python
from improved_pdf_extractor import EnhancedPDFExtractor

# Initialize with ML support
extractor = EnhancedPDFExtractor(
    use_ml=True,
    model_size_limit=200,  # MB
    target_time=30  # seconds for 50 pages
)

# Process PDF
result = extractor.extract_outline_optimized("document.pdf")

print(f"Title: {result['title']}")
print(f"Language: {result['language']}")
print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Headings found: {len(result['outline'])}")
```

This enhanced approach combines the robustness of rule-based methods with the intelligence of small language models, while maintaining fast processing speeds and supporting multiple languages.