# index_data.py
from config import OUTPUT_DIR, INDEX_NAME

import os
import time
import uuid
import re
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =========================
# Load environment variables
# =========================
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# =========================
# Parse transcript with timestamps
# Format:
# [00:01:23] Some text here
# =========================
import re

def parse_transcript(file_path):
    """
    Parse a transcript with timestamps [hh:mm:ss].
    The lines without a timestamp are appended to the previous entry.
    """
    pattern = r"\[(\d{1,2}:\d{2}:\d{2})\]\s*(.*)"
    entries = []
    current_entry = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                # save now if any previous entry exists
                if current_entry:
                    # check for empty entries
                    if current_entry["text"].strip():
                        #print(current_entry)
                        entries.append(current_entry)

                # create a new entry
                current_entry = {
                    "timestamp": match.group(1),
                    "text": match.group(2).replace("♪", "").strip()
                }
            elif current_entry:
                # line without timestamp, save the text
                current_entry["text"] += " " + line.replace("♪", "").strip()
            else:
                # line without timestamp at start: ignore / log
                print(f"Line without timestamp at start: {line}")

    # add the last entry if it exists and has text
    if current_entry and current_entry["text"].strip():
        #print(current_entry)
        entries.append(current_entry)

    return entries


# =========================
# Estimate timestamps per chunk
# =========================
def estimate_timestamps(chunk, entries):
    start_time = "unknown"
    end_time = "unknown"

    for e in entries:
        if e["text"][:40] in chunk:
            start_time = e["timestamp"]
            break

    for e in reversed(entries):
        if e["text"][:40] in chunk:
            end_time = e["timestamp"]
            break

    return start_time, end_time


# =========================
# Text splitter
# =========================
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400,
    chunk_overlap=50
)

# =========================
# Collect documents from all transcripts
# =========================
documents = []

#txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt") and ("_ro_" in f)]

for filename in txt_files:
    file_path = os.path.join(OUTPUT_DIR, filename)
    print(f"Processing {filename}...")

    entries = parse_transcript(file_path)

    if not entries:
        continue

    full_text = "\n".join([e["text"] for e in entries]) #if e["text"].strip() and e["text"] != "[]"])
    chunks = text_splitter.split_text(full_text)

    for chunk in chunks:
        start_time, end_time = estimate_timestamps(chunk, entries)

        video_id = filename.replace(".txt", "").split("_")[-1]  # title format: {video_title}_transcript_{lang}_{video_id}.txt
        #print("video_id:", video_id)

        documents.append({
            "text": chunk,
            "start_time": start_time,
            "end_time": end_time,
            "source_file": filename,
            "video_id": video_id
        })

print(f"Total chunks created from all files: {len(documents)}")

# =========================
# Convert to LangChain Document objects
# =========================
#lc_documents = [
#    Document(
#        page_content=doc["text"],
#        metadata={
#            "text": doc["text"], 
#            "start_time": doc.get("start_time", "unknown"),
#            "end_time": doc.get("end_time", "unknown"),
#            "source_file": doc.get("source_file", "unknown")
#        }
#    )
#    for doc in documents
#]

# =========================
# Embedding model
# =========================
embed = OpenAIEmbeddings(model="text-embedding-3-small")

# =========================
# Pinecone setup
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = INDEX_NAME

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # "eu-west-1"
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


# =========================
# Upload data in batches
# =========================
batch_size = 100

for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
    #batch = lc_documents[i:i + batch_size]
    batch = documents[i:i + batch_size]

    texts = []
    metadatas = []

    for doc in batch:
    
        #text = doc.page_content.strip()
        text = doc.get("text", "").strip()

        texts.append(text)
        #metadatas.append(doc.metadata)
        metadatas.append({
            "text": text, # this is need in order to avoid an error on retriving the data, normally should not be stored in the metadata
            "start_time": doc.get("start_time", "unknown"),
            "end_time": doc.get("end_time", "unknown"),
            "source_file": doc.get("source_file", "unknown"),
            "video_id": doc.get("video_id", "unknown")
        })

    if not texts:
        continue  # nothing to upload in this batch

    embeddings = embed.embed_documents(texts)
    ids = [str(uuid.uuid4()) for _ in texts]

    index.upsert(
        vectors=list(zip(ids, embeddings, metadatas))
    )

# =========================
# or we can upload the data with vector store
# =========================
#vectorstore = PineconeVectorStore.from_documents(
#    documents=documents,
#    embedding=embeddings,
#    index_name=index_name
#)


print("Indexing complete")