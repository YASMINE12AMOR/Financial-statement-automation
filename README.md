# Financial Statement Automation

> **Deep Learning Project — Automatisation des bilans financiers**
> Proposed by the *Pinnacle Insight Team* · In partnership with **BFI (Experts en Technologies Financières)**

An end-to-end pipeline that ingests raw financial documents (PDFs and scanned images), extracts financial statements and their notes, standardizes the output into a structured format, and exposes the data through analytics dashboards and a document-search chatbot.

---

## Overview

Financial statements arrive as heterogeneous, often low-quality PDFs and scans, frequently containing **borderless tables** that conventional OCR tools fail to parse. This project addresses that problem in three stages:

1. **OCR & Layout Detection** — locate and read text from tables (including borderless ones) inside PDFs and images.
2. **NLP & Standardization** — semantically understand accounting sections and convert ~1000 documents into a single standard format.
3. **Analytics & Retrieval** — analyze companies' net results in Power BI and serve the extracted knowledge through an LLM-based document-search chatbot.

The methodology follows the **TDSP (Team Data Science Process)** lifecycle, organized around three pairs of Business Objectives (BO) and Data Science Objectives (DSO).

---

## Objectives

| | Business Objective (BO) | Data Science Objective (DSO) |
|---|---|---|
| **1** | Improve efficiency in financial data processing | Develop a text-extraction / OCR model to accurately extract financial data and content from documents |
| **2** | Enhance recognition of accounting sections and context in financial documents | Apply NLP to understand the semantics and context of extracted information for better categorization and analysis |
| **3** | Standardize output representation for seamless integration | Implement structured data formats, generating output in widely accepted formats (CSV, JSON, XML) |

---

## Architecture

```
                          ┌─────────────────────────┐
                          │   Raw Document (PDF /    │
                          │      scanned image)      │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │   STAGE 1 — Financial Statement Extraction       │
              │                                                  │
              │   • Text extraction → keep pages with a          │
              │     detected financial statement                 │
              │   • Fallback: pytesseract OCR per page;          │
              │     if statement detected → add to PDF writer    │
              └────────────────────────┬────────────────────────┘
                                       │  Financial statement file
              ┌────────────────────────▼────────────────────────┐
              │   STAGE 2 — Layout + OCR (Content Extraction)    │
              │                                                  │
              │   • Layout Element Detection                     │
              │       PaddleDetectionLayoutModel (layoutparser)  │
              │       pretrained on PubLayNet                    │
              │       → detects text, titles, lists,             │
              │         tables, figures                          │
              │   • Text Recognition                             │
              │       PaddleOCR (PP-OCR v4)                       │
              │       DB detection → rectify → CRNN recognition  │
              │   • Post-processing                              │
              │       autoencoder image enhancement +            │
              │       duplicate suppression                      │
              └────────────────────────┬────────────────────────┘
                                       │  Clean tabular text
              ┌────────────────────────▼────────────────────────┐
              │   STAGE 3 — NLP Understanding & Semantic Search  │
              │                                                  │
              │   • Bi-encoder embeddings                        │
              │       SentenceTransformer('all-MiniLM-L6-v2')    │
              │   • Cross-encoder reranking                      │
              │       cross-encoder/msmarco-MiniLM-L-6-v2        │
              │   • Maps raw labels → standard accounting terms  │
              │   • Notes extraction via LLM (GPT-4 API)         │
              └────────────────────────┬────────────────────────┘
                                       │  Standardized output (JSON / CSV / XML)
              ┌────────────────────────▼────────────────────────┐
              │   STAGE 4 — Analytics & Retrieval                │
              │                                                  │
              │   • Power BI dashboards (net results analysis)   │
              │   • LLM document-search chatbot (LLAMA 2)        │
              └──────────────────────────────────────────────────┘
```

---

## Pipeline Stages in Detail

### Stage 1 — Financial Statement Extraction

The script identifies which pages of a document actually contain a financial statement, producing a focused PDF with only the relevant pages. Two scenarios are handled:

- If a financial statement is identified through **direct text extraction**, the page is added to a new PDF.
- If not found via text extraction, the script falls back to **OCR with pytesseract** on each page; pages where OCR detects a financial statement are added to the PDF writer.

### Stage 2 — Layout Detection & OCR

**Layout element detection** uses a model that recognizes layout components (text, titles, lists, tables, figures). It is built on `PaddleDetectionLayoutModel` from **layoutparser**, pretrained on the **PubLayNet** dataset. The model was fine-tuned by:

- Adjusting bounding-box thresholds to improve text precision.
- Tuning the Intersection-over-Union (IoU) threshold to prevent duplicate detections.

**Text recognition** uses **PaddleOCR** in three steps — detect text, rectify alignment, recognize words (DB detection → box rectification → CRNN recognition). It is multilingual and ships ready-to-use models. Fine-tuning included:

- `use_angle_cls` — an angle classifier that straightens skewed images before OCR.
- Using **PP-OCR v4**.
- Including space as an output character in OCR results.

**Post-processing** addresses PaddleOCR's weakness on low-quality scans:

- **Autoencoders** enhance image quality before OCR.
- Custom Python logic suppresses duplicate detections.

This pipeline was chosen after evaluating existing OCR tools (docTR, YOLO, EasyOCR, Camelot, Tesseract), which suffered from issues such as poor JSON handling, manual table labeling, weak element detection, inability to read content inside images, and extraction errors.

### Stage 3 — NLP, Standardization & Notes Extraction

Extracted labels rarely match a clean schema (e.g. OCR noise like *"obilisations Incorporelles"*). A **two-stage semantic search** maps noisy/variant terms onto canonical accounting terms:

1. **Bi-encoder** (`all-MiniLM-L6-v2`) retrieves top-k candidate terms via cosine similarity over sentence embeddings.
2. **Cross-encoder** (`cross-encoder/msmarco-MiniLM-L-6-v2`) reranks the candidates for higher precision.

Financial **notes** (e.g. immobilisations tables, chiffre d'affaires) are extracted and structured into nested JSON using the **GPT-4 API**, producing a standardized representation across ~1000 documents.

### Stage 4 — Analytics & Chatbot

- **Power BI** consumes the standardized output to analyze companies' net results and build interactive dashboards.
- **`text.py`** — Streamlit interface that allows users to upload a financial document (PDF) and ask questions to get an explanation of its content, powered by LLAMA 2 via Ollama.
- **`pdfextraction.py`** — Flask REST API exposing a conversational interface over financial documents using LangChain memory chains and the OpenAI API.

---

## Evaluation Metrics

The OCR pipeline was benchmarked on two axes (measured on 30 CSV files, limited by Colab RAM):

| Metric | Result |
|---|---|
| Total CSV lines processed | 1,079 |
| Lines containing duplicates | 27 |
| Error margin | ≈ 0.025 |
| **Accuracy** | **≈ 0.975** |
| Execution time | ≈ 5.79 s |

---

## Tech Stack

| Layer | Tools |
|---|---|
| OCR & Layout | PaddleOCR (PP-OCR v4), layoutparser (PaddleDetectionLayoutModel, PubLayNet), pytesseract |
| Deep Learning | TensorFlow (autoencoders for image enhancement) |
| NLP / Embeddings | Hugging Face, Sentence Transformers, `all-MiniLM-L6-v2`, cross-encoder reranking, Semantic Search |
| LLMs | GPT-4 API (notes extraction), LLAMA 2 (chatbot), OpenAI API (conversational API) |
| Chatbot / API | Streamlit (`text.py`), Flask + LangChain (`pdfextraction.py`), Ollama, FAISS |
| Analytics | Power BI |
| Output formats | CSV, JSON, XML |

---

## Project Structure

```
Financial-statement-automation/
├── pdfextraction.py       # Flask API — conversational interface via LangChain + OpenAI
├── text.py                # Streamlit app — upload a financial PDF and explain its content via LLAMA 2
├── llama/                 # Submodule: Meta's LLaMA reference implementation
├── llama.cpp/             # Submodule: C++ LLM inference engine (ggerganov/llama.cpp)
├── Reports/               # Output reports directory
├── .env                   # Environment variables (not tracked — see below)
└── .gitignore
```

### Environment Variables (`.env`)

Copy `.env` and fill in your values before running:

```
OPENAI_API_KEY=          # Required by pdfextraction.py
FLASK_SECRET_KEY=        # Flask session security
DATABASE_URL=            # SQLAlchemy connection string (default: sqlite:///app.db)
HUGGINGFACE_API_TOKEN=   # Optional — for private HF models
OLLAMA_BASE_URL=         # Ollama server URL (default: http://localhost:11434)
```

### Running the apps

```bash
# Streamlit PDF chatbot (text.py) — requires Ollama running with llama2
streamlit run text.py

# Flask conversational API (pdfextraction.py) — requires OPENAI_API_KEY
python pdfextraction.py
```

---

## Process

The project follows the **TDSP** lifecycle:

- **Phase 2 — Data Understanding:** standardizing raw model output into structured JSON.
- **Phase 3 — Adequate Model:** selecting and fine-tuning the layout + OCR + NLP stack.
- **Evaluation:** accuracy and execution-time benchmarking against existing OCR baselines.

---

## Conclusion

As AI and machine learning mature, integrating them into financial-statement automation opens the door to stronger predictive analytics and richer financial intelligence — turning unstructured, low-quality documents into standardized, queryable, analyzable data.

---

## Team

**Pinnacle Insight Team:** Nadine Elleuch · Siwar Najjar · Yasmine Amor · Islem Maiti · Abir Zahra · Arij Afaya
**Supervisors:** Wiem Trabelsi · Sarra Zouari · Mohamed Aziz Kasseb
