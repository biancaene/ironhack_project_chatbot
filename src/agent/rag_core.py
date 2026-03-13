# rag_core.py
import os
import time
from dotenv import load_dotenv, find_dotenv

from src.agent.config import INDEX_NAME, LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Embeddings + Pinecone
embed = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embed,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Prompt RAG
rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about YouTube videos.

Use only the context below.

The sources are ordered by relevance.
Source 1 is the most relevant.

RULES:
- Prefer Source 1 when possible
- Use other sources only if needed
- If the answer is not in the sources, say you don't know

IMPORTANT:
- Always include the VIDEO LINK and TIMESTAMP if available.
- Do not remove, hide, or paraphrase the link.
- Format each answer as:    

[TIMESTAMP] [VIDEO_URL]
Transcript: <relevant text>

Context:
{context}

Question: {question}
"""
)

def time_to_seconds(t: str) -> int:
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def format_docs(docs):
    formatted = []
    for i, d in enumerate(docs):
        start_time = d.metadata.get("start_time", "unknown")
        end_time = d.metadata.get("end_time", "unknown")
        source_file = d.metadata.get("source_file", "unknown")
        video_id = d.metadata.get("video_id", "unknown")

        label = f"Source {i+1}"
        if i == 0:
            label += " - MOST RELEVANT"

        formatted.append(
            f"""
                [{label}]
                Video ID: {video_id}
                Time: {start_time} - {end_time}
                Source file: {source_file}

                Transcript:
                {d.page_content}
            """
        )
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

def run_rag(query: str):
    """RAG: întoarce answer + segmente video (din retriever)."""
    docs = retriever.invoke(query)

    segments = []
    for d in docs:
        start = d.metadata.get("start_time")
        end = d.metadata.get("end_time")
        video_id = d.metadata.get("video_id")
        seconds = time_to_seconds(start)

        segments.append({
            "video_id": video_id,
            "start_time": start,
            "end_time": end,
            "seconds": seconds,
            "watch_link": f"https://youtube.com/watch?v={video_id}&t={seconds}s",
            #"embed_link": f"https://www.youtube.com/embed/{video_id}?start={seconds}",
            "embed_link": f"https://www.youtube.com/embed/{video_id}?start={seconds}&end={time_to_seconds(end) if end else ''}",
            "text": d.page_content
        })

    answer = rag_chain.invoke(query)

    return {
        "answer": answer,
        "segments": segments
    }
