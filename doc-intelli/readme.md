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
