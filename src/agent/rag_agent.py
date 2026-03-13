# rag_agent.py
import os
from unittest import result
from dotenv import load_dotenv, find_dotenv

from langchain_core.tools import tool
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.agents import initialize_agent, AgentType

from src.agent.rag_core import run_rag
from src.agent.rag_core import llm

from src.agent.video_player import get_stream_url

import yfinance as yf

_ = load_dotenv(find_dotenv())


# ---------------- TOOL RAG ----------------

@tool
def rag_qa(query: str) -> dict:
    """
    Answer questions about YouTube videos.
    Returns:
        answer: text
        segments: list of video segments
    """
    return run_rag(query)


# ---------------- TOOL: Stock Ticker Lookup ----------------

@tool
def get_stock_ticker(company_name: str) -> str:
    """
    Return the stock ticker symbol and current price for a given company name.
    Example:
        Input: "Tesla"
        Output: "TSLA - Tesla Inc, Price: 250.3"
    """
    try:
        results = yf.Search(company_name, max_results=5)
        quotes = results.quotes

        if not quotes:
            return f"No ticker found for {company_name}"

        best = quotes[0]
        symbol = best.get("symbol", "")
        name = best.get("shortName", "")
        
        ticker = yf.Ticker(symbol)
        price = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice", "N/A")

        return f"{symbol} - {name}, Price: {price}"

    except Exception as e:
        return f"Error retrieving ticker: {str(e)}"

# ---------------- MEMORY ----------------

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)


# ---------------- AGENT ----------------

agent_executor = initialize_agent(
    llm=llm,
    tools=[rag_qa, get_stock_ticker],
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    agent_kwargs={
        "system_message": (
            "You are a multimodal AI assistant. Decide which tool to use: "
            "rag_qa for video questions, "
            "get_stock_ticker for company ticker lookup. "
            #"You are an assistant answering questions about YouTube videos. "
            #"Always use the rag_qa tool to retrieve video segments and transcripts. "
            #"Return the answer clearly and reference the timestamps."
        )
    },
    return_intermediate_steps=True
)


# ---------------- CHAT ----------------

def chat_with_agent(user_input: str) -> dict:
    """
    Conversational answer from agent.
    If the agent used rag_qa → return video segments.
    If the agent used get_stock_ticker → return no segments.
    """

    # run agent
    #agent_answer = agent_executor.run(user_input)
    result = agent_executor.invoke({"input": user_input})

    agent_answer = result["output"]
    steps = result["intermediate_steps"]
    print("agent_answer = ", agent_answer)
    print("STEPS:", steps)
 
    # detect which tool was used
    # LangChain agents include tool names in the output text
    used_rag = False
    used_ticker = False
    rag_result = None

    for action, observation in steps:
        print("TOOL USED:", action.tool)

        if action.tool == "rag_qa":
            used_rag = True
            rag_result = observation

        elif action.tool == "get_stock_ticker":
            used_ticker = True
    
    #print("used_rag = ", used_rag)
    #print("used_ticker = ", used_ticker)

    # if RAG was used -> get segments
    if used_rag:
        # video segments from RAG
        #rag_result = run_rag(user_input)
        print("rag_result = ", rag_result)

        # add stream_url for each segment, needed to jump to a certain timestamp
        #for seg in rag_result["segments"]:
            # stream direct (HLS/MP4)
        #    try:
        #        media_url = get_stream_url(seg["watch_link"])

                # HTML5 timestamp slicing
        #        seg["stream_url"] = f"{media_url}#t={seg['seconds']}"
        #    except Exception as e:
        #        print(f"Could not get stream URL for {seg['watch_link']}: {e}")
        #        seg["stream_url"] = None

        return {
            "answer": agent_answer,
            "segments": rag_result["segments"],
            "intermediate_steps": steps
        }

    # if ticker tool was used -> no video segments
    if used_ticker:
        return {
            "answer": agent_answer,
            "segments": [],
            "intermediate_steps": steps
        }

    # if no tool was used → default fallback
    return {
        "answer": agent_answer,
        "segments": [],
        "intermediate_steps": steps
    }
