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
