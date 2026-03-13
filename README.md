**IronHack AI Engineering Bootcamp – January 2026**  

# Project III | Business Case: Building a Multimodal AI ChatBot for YouTube Video QA

[Project in GitHub](https://github.com/biancaene/)

---

## Project Goal

The goal of this project is to build a multimodal AI chatbot capable of understanding and answering natural‑language questions about YouTube videos. By combining text processing, speech recognition, and retrieval‑augmented generation (RAG), the system converts video content into searchable transcripts, stores them in a vector database, and enables users to query videos through text or voice. 

---

## Project Overview
This project implements an end‑to‑end multimodal RAG system that answers user questions about YouTube videos by transforming raw video content into searchable knowledge. The pipeline automates video retrieval, audio/transcript extraction, embedding generation, vector storage, and intelligent querying through a LangChain‑powered agent and a Streamlit UI.

**Part 1: YouTube Data Ingestion**
- The system begins by collecting all video URLs from a target YouTube channel using `channel.py`. These URLs serve as the entry point for downstream processing.
  
**Part 2: Video & Transcript Extraction**
- For each video, the pipeline downloads the video track and the official YouTube transcript via `download_video.py` and `download_transcript.py`. This produces the raw text and video needed for further analysis.
    
**Part 3: Indexing & Embedding Generation**
- `index_data.py` processes the extracted text by chunking it into manageable segments and generating vector embeddings. These embeddings are stored in a Pinecone vector database, enabling efficient semantic search. 

**Part 4: Retrieval‑Augmented Generation (RAG) Core**
- `rag_core.py` handles the retrieval logic: given a user query, it searches Pinecone for the most relevant video chunks and prepares context for answer generation.

**Part 5: Intelligent Agent Layer**
- `rag_agent.py` wraps the RAG pipeline inside a LangChain agent equipped with tools and memory. The agent orchestrates retrieval, reasoning, and response generation, enabling natural conversational interaction.

**Part 6: User Interface**
- The final application is delivered through `app_streamlit.py`, a Streamlit‑based UI that allows users to ask questions, view retrieved context, and interact with the chatbot through a clean, intuitive interface.

---

## Project Evaluation
- **Retrieval Quality**: Accuracy and relevance of video‑based answers returned by the RAG pipeline.

- **System Performance**: End‑to‑end latency across ingestion, retrieval, and response generation.

- **Agent Behavior**: Correct use of tools, memory handling, and conversational coherence.

- **UI/UX**: Ease of use, clarity, and stability of the Streamlit interface.

- **Pipeline Reliability**: Successful execution of each stage (URL extraction, downloads, indexing, embeddings, Pinecone queries).

- **Overall Accuracy**: Faithfulness of answers compared to the original video content.

---

## Project Results
- Built a complete RAG pipeline that processes YouTube videos into searchable embeddings.

- Implemented a LangChain agent that retrieves relevant context and generates accurate answers.

- Delivered a functional Streamlit app enabling smooth, natural querying of video content.

---

## Project Content

The project is organized into modular Python components, each responsible for a specific stage of the YouTube‑to‑RAG pipeline:

- `src/agent/`

  - `channel.py`: Retrieves all video URLs from a given YouTube channel.

  - `download_video.py` / `download_transcript.py`: Extracts audio or transcripts from each video.

  - `generate_transcript.py`: For the videos that do not have a transcript available, OpenAI's Whisper API is used to transcript the videos.

  - `index_data.py`: Chunks text, generates embeddings, and indexes them into the vector database.

  - Pinecone Vector DB: Stores embeddings for fast semantic search and retrieval.

  - `rag_core.py`: Implements the core RAG retrieval logic using stored embeddings.

  - `rag_agent.py`: Wraps the RAG system inside a LangChain agent with tools and memory.

- `src/eval`
  
  - `evaluate_agent.py`: Evaluation is powered by LangSmith, using LLM-as-a-judge scoring across three automated metrics — correctness, relevance, and groundedness — alongside a custom tool-use evaluator that validates proper agent tool inputs.  

- `src/deployment`
  
  - `app_streamlit.py`: Provides the user interface for querying videos through a Streamlit app.

- `downloads/`
  - The place were the videos and transcripts are downloaded.

- `presentation/`
  - `presentation.pptx`: presentation to exhibit results, methodology, and findings.

- `requirements/`
  - The `requirements.txt` file can be used to install the environment needed to run the notebooks:

It is recommended to use a **virtual environment**.

---

## Environment Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```
