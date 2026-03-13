# config.py
# YouTube channel ID to index
CHANNEL_ID = "UCWgk87pKcNHragGsV2WIh5w"

# location for downloaded videos, transcripts, etc
OUTPUT_DIR = "downloads"

# number of latest videos to fetch and index
VIDEO_LIMIT = 30

DELAY_MIN = 30
DELAY_MAX = 60

# config for Pinecone
#INDEX_NAME = "test-en-index"
INDEX_NAME = "test-ro-index"

# config for LangSmith / LangChain tracing
LANGCHAIN_PROJECT="chatbot"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"