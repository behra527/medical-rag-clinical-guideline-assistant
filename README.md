
# Medical RAG — Clinical Guideline Assistant

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-FTS5-003B57?logo=sqlite&logoColor=white)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6366F1)
![RAG](https://img.shields.io/badge/Architecture-Hybrid_RAG-8B5CF6)
![Status](https://img.shields.io/badge/Project-Research-blue)

## Overview

Medical RAG is an AI-powered clinical guideline question-answering system built using **Retrieval-Augmented Generation (RAG)**.

It retrieves relevant information from indexed clinical guideline documents and generates concise, evidence-grounded responses using Large Language Models (LLMs).

The system focuses on **answer accuracy, hallucination reduction, source traceability, and evidence verification**.

## Key Features

- **Document Indexing:** Upload and process clinical guideline PDFs.
- **Intelligent Chunking:** Split documents into overlapping 190-word chunks.
- **Hybrid Retrieval:** Combine semantic similarity, SQLite FTS5, and reranking.
- **LLM Integration:** Generate answers through OpenRouter.
- **Prompt Engineering:** Produce concise and context-aware responses.
- **Evidence Validation:** Apply a 0.40 similarity threshold.
- **Claim Verification:** Check generated claims against retrieved evidence.
- **Source Citations:** Include document page numbers and section references.
- **Hallucination Reduction:** Abstain when evidence is insufficient.
- **Streaming Responses:** Support real-time answer streaming using SSE.

## System Architecture

The system consists of two connected pipelines: document indexing and question answering.

```mermaid
flowchart TB
    subgraph INDEX["DOCUMENT INDEXING PIPELINE"]
        direction LR
        A["PDF Upload / Import"] --> B["FastAPI Validation"]
        B --> C["PyMuPDF Extraction"]
        C --> D["190-Word Chunking"]
        D --> E["OpenRouter Embeddings"]
        E --> F[("SQLite + FTS5")]
    end

    subgraph QUERY["QUESTION ANSWERING PIPELINE"]
        direction LR
        G["User Question"] --> H["FastAPI /chat"]
        H --> I["Query Rewrite / Filters"]
        I --> J["Hybrid Retrieval + Reranking"]
        J --> K["Evidence Validation"]
        K --> L["LLM + Claim Checking"]
        L --> M["Cited Answer / Abstention"]
    end

    F --> J

    style INDEX fill:#eff6ff,stroke:#2563eb,color:#1e3a8a
    style QUERY fill:#f0fdf4,stroke:#16a34a,color:#14532d
```

## Answer Verification Workflow

To improve reliability, the system validates retrieved evidence before producing a final response.

```mermaid
flowchart TD
    A["User Question"] --> B["Hybrid Retrieval"]
    B --> C{"Similarity >= 0.40?"}

    C -->|No| D["Abstain"]
    C -->|Yes| E["OpenRouter LLM"]

    E --> F["Concise Prompt"]
    F --> G["Claim Verification"]

    G --> H{"Claims Supported?"}

    H -->|No| D
    H -->|Yes| I["Attach Page + Section Citations"]

    I --> J["Final Evidence-Grounded Answer"]

    style D fill:#fee2e2,stroke:#dc2626,color:#991b1b
    style J fill:#dcfce7,stroke:#16a34a,color:#166534
    style C fill:#fef3c7,stroke:#d97706,color:#92400e
    style H fill:#fef3c7,stroke:#d97706,color:#92400e
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.11 |
| Backend | FastAPI, Uvicorn |
| LLM Provider | OpenRouter |
| Chat Model | GPT-4o Mini |
| Embedding Model | text-embedding-3-small |
| Database | SQLite |
| Full-Text Search | SQLite FTS5 |
| PDF Processing | PyMuPDF |
| Similarity Search | NumPy, Cosine Similarity |
| Data Validation | Pydantic |
| API Communication | HTTPX |
| Frontend | HTML, CSS, JavaScript |
| Response Streaming | Server-Sent Events |
| Testing | Pytest |

## Core AI Techniques

### 1. Retrieval-Augmented Generation (RAG)
Combines document retrieval and LLM-based generation to answer questions using relevant clinical guideline content.

### 2. Hybrid Search
Uses semantic vector similarity and keyword-based full-text search to retrieve relevant information.

### 3. Reranking
Reorders retrieved passages to prioritize the most relevant evidence.

### 4. Prompt Engineering
Uses optimized system prompts to generate clear, concise, and relevant responses.

### 5. Hallucination Prevention
Applies similarity gating, claim checking, and an abstention mechanism to reduce unsupported answers.

### 6. Citation-Aware Generation
Attaches document page numbers and section references to improve answer traceability.

## Project Objectives

- Develop an evidence-grounded medical question-answering system.
- Improve clinical document retrieval using hybrid search.
- Minimize unsupported LLM-generated information.
- Improve response clarity through prompt optimization.
- Provide transparent and traceable source citations.

## Medical Disclaimer

This project is intended for educational and research purposes.

It is not a substitute for professional medical advice, diagnosis, or treatment. All AI-generated information should be reviewed by qualified healthcare professionals before clinical use.

---

**Medical RAG — Building reliable AI systems through retrieval, verification, and evidence-grounded generation.**
