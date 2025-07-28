# PDF Outline Extractor

## Project Description
This solution extracts structured outlines (titles and hierarchical headings) from a collection of PDF documents. It is designed to work generically for any set of PDFs, using font and layout cues to identify document structure. The tool is fully containerized for easy, reproducible, and offline use.

---

## Features
- **Automatic Outline Extraction:** Identifies titles and section headings from PDFs using font size, boldness, and text patterns.
- **Domain Agnostic:** Works with research papers, books, reports, and more.
- **Offline & Reproducible:** All dependencies are included in the Docker image; no internet required at runtime.
- **Simple Output:** Produces a JSON outline for each PDF, including the document title and a list of headings with hierarchy and page numbers.

---

## Input Specification
- Place your PDF files in the `input/` directory (or mount a directory as `/app/input` in Docker).
- The script will process all `.pdf` files in this directory.

**Example directory structure:**
```
input/
  doc1.pdf
  doc2.pdf
  ...
```

---

## Output Specification
- For each input PDF, a JSON file is created in the `output/` directory (or `/app/output` in Docker).
- Each output file is named after the input PDF (e.g., `doc1.json` for `doc1.pdf`).

**Example output structure:**
```
output/
  doc1.json
  doc2.json
  ...
```

**Example output JSON:**
```
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    ...
  ]
}
```

---

## How It Works (Architecture)
1. **PDF Parsing:** Uses PyMuPDF to extract text spans, font sizes, and layout information from each page.
2. **Heading Detection:** Applies rules based on font size, boldness, and text patterns to identify headings and their hierarchy.
3. **Outline Construction:** Builds a structured outline with heading levels (H1, H2, etc.), text, and page numbers.
4. **Output Generation:** Saves the outline as a JSON file for each PDF.

---

## Thought Process & Approach

### Motivation
Extracting a structured outline from arbitrary PDFs is challenging due to the lack of standard formatting across documents. Most PDFs do not contain explicit metadata for headings or structure, so the approach must infer document structure from visual and textual cues. The goal is to create a robust, domain-agnostic solution that works for research papers, books, reports, and more.

### Problem Analysis
- **PDFs are visually formatted:** Unlike HTML or DOCX, PDFs are designed for visual rendering, not semantic structure. Headings, paragraphs, and lists are not explicitly marked.
- **Diversity of layouts:** Academic papers, books, and reports use different conventions for headings (numbered, unnumbered, all-caps, bold, etc.).
- **Noisy content:** Footers, headers, page numbers, and watermarks can be mistaken for headings.

### Approach Overview
The solution is based on the hypothesis that headings are visually distinct from body text. The extractor uses a combination of font metrics, text patterns, and layout cues to infer structure. The process is as follows:

1. **Text Span Extraction:**
   - For each page, extract all text spans with their font size, font family, style (bold/italic), and bounding box.
   - Example: A span might be {"text": "1. Introduction", "size": 18.0, "font": "Times-Bold", "flags": 16, ...}

2. **Text Cleaning and Normalization:**
   - Remove extra whitespace, normalize Unicode, and strip out non-informative characters.
   - Rationale: Clean text improves pattern matching and reduces false positives.
   - Example: "  1.   Introduction\n" → "1. Introduction"

3. **Heuristic Heading Detection:**
   - **Font Size:** Headings are usually larger than body text. The extractor computes the average font size and flags spans significantly larger as potential headings.
   - **Boldness:** Bold or all-caps text is more likely to be a heading. The extractor checks font flags and text casing.
   - **Text Patterns:** Numbering (e.g., "2.1 Methods"), title case, and keywords ("Chapter", "Section") are strong indicators.
   - **Font Family:** A change in font family (e.g., from serif to sans-serif) often marks a heading.
   - **Position:** Headings often appear at the top of a page or section. The extractor considers the vertical position (y-coordinate) for title detection.
   - **Negative Heuristics:** Spans that are too short, mostly numeric, or match common footer/header patterns ("Page 1 of 10") are excluded.
   - **Scoring System:** Each cue contributes to a score; only spans above a threshold are considered headings.
   - **Example:**
     - "2.1 Methods" (size=16, bold, y=120) → Score: 6 (heading)
     - "Page 2 of 10" (size=10, not bold, y=800) → Score: 0 (ignored)

4. **Deduplication and Sorting:**
   - Headings with identical text on the same page are deduplicated.
   - Headings are sorted by page number and vertical position to preserve document order.

5. **Hierarchy Assignment:**
   - **Font Size Grouping:** Unique font sizes among headings are sorted descending; the largest is mapped to H1, next to H2, etc.
   - **Pattern Overrides:** Numbering patterns ("1.", "1.1", "1.1.1") override font-based levels to better reflect logical structure.
   - **Fallbacks:** If font sizes are too close, or the document uses only one size, numbering and indentation are used to infer hierarchy.
   - **Example:**
     - "1. Introduction" (size=18) → H1
     - "1.1 Background" (size=16) → H2
     - "1.1.1 Details" (size=14) → H3

6. **Title Extraction:**
   - The extractor looks for the largest or boldest text near the top of the first page, skipping common headers.
   - If multiple lines are close in size and position, they are concatenated.
   - Fallback: If no clear title, use the largest text span on the first page.
   - **Example:**
     - "A Study on PDF Parsing" (size=24, y=80) → Title
     - "Research Report" (size=20, y=120) → Appended if close

7. **Output Construction:**
   - The outline is output as a list of heading objects with level, text, and page number.
   - The title is included as a separate field.

### Challenges and Edge Cases
- **Inconsistent Formatting:** Some documents use the same font size for headings and body text. The extractor relies more on numbering and boldness in these cases.
- **No Numbering:** For narrative documents (e.g., novels), headings may not be numbered. The extractor uses title case, font, and position.
- **False Positives:** Footers, headers, and watermarks are filtered using negative heuristics and position.
- **Multi-line Titles:** Titles split across multiple lines are combined if they are close in size and position.
- **Sparse Documents:** For very short or single-page documents, the extractor falls back to the largest or boldest text as the title/heading.

### Alternative Approaches Considered
- **Machine Learning:** Training a model to classify headings was considered, but would require labeled data and may not generalize across domains.
- **PDF Metadata:** Some PDFs contain bookmarks or outlines, but these are rare and inconsistent.
- **Template Matching:** Hardcoding for specific document types was rejected to keep the solution general.

### Adaptability
- The extractor is designed to work with a wide range of document types, from academic papers to business reports and books.
- Heuristics can be tuned or extended for specific domains if needed.
- The code is modular, allowing for easy integration of new rules or ML-based enhancements in the future.

### Summary
The solution leverages a combination of visual cues (font size, boldness, font family, position) and textual patterns (numbering, case, keywords) to infer document structure. By combining these heuristics, it achieves high accuracy in extracting outlines from unstructured PDFs, producing a clean, hierarchical JSON output for downstream use. The approach is robust, explainable, and adaptable to new document types and layouts.

---

## How to Build and Run (Docker)

### 1. Build the Docker Image
```sh
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### 2. Prepare Input/Output Folders
- Place your PDFs in an `input/` folder in your working directory.
- Create an empty `output/` folder (or let Docker create it).

### 3. Run the Solution
```sh
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-outline-extractor:latest
```
- On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.
- The script will process all PDFs in `/app/input` and write JSON outlines to `/app/output`.

---

## Troubleshooting
- If you see errors about missing files, check your input/output folder mounts.
- If Docker says `python: can't open file ...`, check the script name in the Dockerfile.
- For PDF parsing issues, ensure your PDFs are not encrypted or corrupted.

---

## Contact
For questions or issues, please drop a mail at soumenkumar9503@gmail.com or b22cs058@iitj.ac.in

## 🔒 Access Instructions for Judges

This is a private repository submitted for the Adobe-India-Hackathon25.

Please ensure you're logged in with the GitHub account you were invited with(jhaaj08 and rbabbar-adobe). 
If you need access again, contact: soumenkumar9503@gmail.com

