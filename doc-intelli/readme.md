# Intelligent Document Analyst

## Challenge Brief
This solution is designed for the Adobe Document Intelligence Challenge. The goal is to build a system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

### Key Features
- **Generic:** Works with any domain (research, business, education, etc.)
- **Persona-aware:** Adapts extraction and ranking to the provided persona and task
- **Offline:** Runs fully offline after Docker build (no internet required at runtime)
- **CPU-only:** All processing is CPU-based, with model size <1GB
- **Fast:** Processes 3-5 documents in under 60 seconds

---

## Input Specification
- **Document Collection:** 3-10 related PDFs (placed in the appropriate input folder)
- **Persona Definition:** Role description with expertise and focus areas (provided in JSON config)
- **Job-to-be-Done:** Concrete task the persona needs to accomplish (provided in JSON config)

### Example Input Structure
```
input/
  Collection 1/
    PDFs/
      doc1.pdf
      doc2.pdf
      ...
    challenge1b_input.json
  Collection 2/
    PDFs/
    challenge1b_input.json
  Collection 3/
    PDFs/
    challenge1b_input.json
```

#### Example `challenge1b_input.json`
```
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",
    "description": "France Travel"
  },
  "documents": [
    {"filename": "doc1.pdf", "title": "Doc 1"},
    ...
  ],
  "persona": {"role": "Travel Planner"},
  "job_to_be_done": {"task": "Plan a trip of 4 days for a group of 10 college friends."}
}
```

---

## Output Specification
- **Output JSON** is saved as:
  - `output/<challenge_id>/final_output.json`
- **Output Structure:**
```
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "..."
  },
  "extracted_sections": [
    {
      "document": "...",
      "section_title": "...",
      "importance_rank": 1,
      "page_number": 3
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "...",
      "refined_text": "...",
      "page_number": 3
    },
    ...
  ]
}
```

---

## How It Works (Architecture)
1. **PDF Parsing:** Extracts headings and content from each PDF using font, layout, and text cues.
2. **Semantic Ranking:** Uses a compact transformer model (MiniLM) to rank sections by relevance to the persona and job-to-be-done.
3. **Output Generation:** Produces a structured JSON with the most relevant sections and their content, tailored to the persona/task.
4. **Offline Model Caching:** The model is pre-downloaded and cached in the Docker image for offline use.

---

## Thought Process & Approach

### Motivation
The challenge is to extract and prioritize the most relevant sections from a diverse set of PDFs, tailored to a specific persona and their job-to-be-done. PDFs lack explicit semantic structure, and the definition of "relevance" changes with the persona and task. The solution must be generic, robust, and explainable.

### Problem Analysis
- **PDFs are visually formatted:** Unlike HTML or DOCX, PDFs are designed for visual rendering, not semantic structure. Headings, paragraphs, and lists are not explicitly marked.
- **Diversity of layouts:** Academic papers, business reports, and educational materials use different conventions for headings and sectioning.
- **Dynamic relevance:** What matters to a business analyst is different from what matters to a student or researcher.

### Approach Overview
The system is built on the following principles:

1. **Visual Structure Extraction:**
   - Extract all text spans from each PDF, capturing font size, font family, style (bold/italic), and position.
   - Use heuristics (font size, boldness, numbering, title case, font family) to identify headings and their hierarchy.
   - Example: "2.1 Methods" (size=16, bold) is likely a section heading; "Page 2 of 10" (size=10) is ignored.

2. **Content Segmentation:**
   - For each detected heading, extract the text content between it and the next heading, spanning multiple pages if needed.
   - This provides context for each section, not just the heading text.

3. **Semantic Ranking with Persona Awareness:**
   - Use a compact transformer model (MiniLM) to embed both the persona/task and each section (heading + content).
   - Compute semantic similarity between the persona/task and each section to rank them by relevance.
   - Example: For persona "Travel Planner" and job "Plan a trip for 10 friends," sections about group activities or itineraries will rank higher.

4. **Output Construction:**
   - Select the top N most relevant sections across all documents.
   - Output includes the document, section title, page number, and a content preview for each top section.

### Key Design Decisions & Rationale
- **Heuristic Heading Detection:**
  - Chosen for its generality and explainability. Font and text cues are reliable across most document types.
  - Negative heuristics (e.g., filtering out short, numeric, or repetitive footer/header text) reduce false positives.
- **Semantic Embedding for Ranking:**
  - Allows flexible, context-aware relevance scoring that adapts to any persona or task without retraining.
  - Outperforms keyword matching, especially for nuanced or domain-specific queries.
- **Offline Model Caching:**
  - Ensures the solution is fully offline and reproducible, with no runtime downloads.
- **Persona/Task Adaptability:**
  - The system is agnostic to the domain; changing the persona/task in the config instantly adapts the ranking.

### Challenges and Edge Cases
- **Inconsistent Formatting:** Some documents use the same font size for headings and body text. The extractor relies more on numbering and boldness in these cases.
- **No Numbering:** For narrative or educational documents, headings may not be numbered. The extractor uses title case, font, and position.
- **False Positives:** Footers, headers, and watermarks are filtered using negative heuristics and position.
- **Sparse Documents:** For very short or single-page documents, the extractor falls back to the largest or boldest text as the title/heading.
- **Multi-document Ranking:** Ensures that the most relevant sections are selected across all documents, not just within each document.


### Adaptability & Extensibility
- The system is modular: heuristics and models can be tuned or replaced for specific domains.
- New personas or tasks can be supported by simply updating the config JSON.
- The codebase is structured for easy integration of new rules or ML-based enhancements in the future.

### Example
- **Persona:** "Investment Analyst"
- **Job:** "Analyze revenue trends and R&D investments"
- **Result:** Sections about financials, R&D, and market analysis are ranked highest, regardless of their position or explicit keywords.

### Summary
This approach combines robust visual heuristics for structure extraction with powerful semantic ranking for relevance, all tailored to the user's persona and task. The result is a flexible, explainable, and high-accuracy system for intelligent document analysis across any domain.

---

## How to Build and Run (Docker)

### 1. Build the Docker Image
```sh
docker build --platform linux/amd64 -t doc-intelli .
```
- This will install all dependencies and pre-download the required model for offline use.

### 2. Prepare Input/Output Folders
- Place your input PDFs and `challenge1b_input.json` in the appropriate `input/Collection X/` folders.
- Ensure you have an `output/` folder (or let Docker create it).

### 3. Run the Solution
```sh
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none doct-intelli
```
- On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.
- The script will prompt you to select the test collection (1, 2, or 3).
- Output will be saved in `output/<challenge_id>/final_output.json`.

---

## Constraints & Notes
- **CPU Only:** No GPU required or used.
- **Model Size:** <200MB (MiniLM)
- **No Internet:** All models and dependencies are pre-cached; no downloads at runtime.
- **Processing Time:** <60 seconds for 3-5 documents (typical case)
- **Generic:** Works for any persona/task/document type as long as the input format is followed.

---

## Example Use Cases
- **Academic Research:**
  - Persona: PhD Researcher in Computational Biology
  - Job: "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
- **Business Analysis:**
  - Persona: Investment Analyst
  - Job: "Analyze revenue trends, R&D investments, and market positioning strategies"
- **Educational Content:**
  - Persona: Undergraduate Chemistry Student
  - Job: "Identify key concepts and mechanisms for exam preparation on reaction kinetics"

---

## Troubleshooting
- If you see errors about missing files, check your input/output folder mounts.
- If Docker says `python: can't open file ...`, check the script name in the Dockerfile.
- For model download/caching issues, ensure the Docker build completes without errors.

---

## Contact
For questions or issues, please drop a mail at soumenkumar9503@gmail.com or b22cs058@iitj.ac.in

## 🔒 Access Instructions for Judges

This is a private repository submitted for the Adobe-India-Hackathon25.

Please ensure you're logged in with the GitHub account you were invited with(jhaaj08 and rbabbar-adobe). 
If you need access again, contact: soumenkumar9503@gmail.com