# app_streamlit.py
import streamlit as st
import openai
from openai import OpenAI

import sys
import os
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agent.rag_agent import chat_with_agent
from src.eval.evaluate_agent import evaluate_live

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


# -----------------------------
# Transcription function (Whisper)
# -----------------------------
def transcribe_audio(audio_file):
    """Return text transcription using Whisper (OpenAI v1+)."""
    if audio_file is None:
        return ""

    audio_file.seek(0)

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_file, "audio/wav")
    )

    return transcript.text


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎥 Multimodal AI ChatBot for YouTube Video QA")

# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_embed" not in st.session_state:
    st.session_state.current_embed = None

if "user_msg" not in st.session_state:
    st.session_state.user_msg = ""

if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = ""

if "typed_text" not in st.session_state:
    st.session_state.typed_text = ""

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# -----------------------------
# Layout: Chat + Video
# -----------------------------
col_chat, col_video = st.columns([1, 1])


# -----------------------------
# CHAT (text + voice)
# -----------------------------
with col_chat:
    st.markdown("## 💬 Ask a Question")

    # --- TEXT INPUT ---
    typed = st.text_input(
        "💬 Type your question",
        placeholder="Ex: What does the speaker say about topic X?",
        key=f"typed_{st.session_state.input_key}"
    )
    st.session_state.typed_text = typed

    # --- AUDIO INPUT (cu key dinamic pentru resetare) ---
    st.markdown("## 🎙️ Or ask by voice")
    audio_data = st.audio_input("", key=f"audio_{st.session_state.audio_key}")

    if audio_data is not None:
        st.info("Transcribing audio…")
        st.session_state.audio_transcript = transcribe_audio(audio_data)
        st.success(f"Transcribed: {st.session_state.audio_transcript}")

    # --- SEND BUTTON ---
    if st.button("🚀 Send", use_container_width=True):

        # Audio has priority if freshly recorded, else use typed text
        final_msg = (
            st.session_state.audio_transcript.strip()
            if st.session_state.audio_transcript.strip()
            else st.session_state.typed_text.strip()
        )

        if final_msg.strip():
            try:
                result = chat_with_agent(final_msg)
                steps  = result.get("intermediate_steps", [])
                scores = evaluate_live(final_msg, result["answer"], steps)

                st.session_state.history.append(("user", final_msg))
                st.session_state.history.append(("assistant", result, scores))
            except Exception as e:
                st.error(f"❌ Error: {e}")

        # reset
        st.session_state.typed_text = ""
        st.session_state.audio_transcript = ""
        st.session_state.audio_key += 1
        st.session_state.input_key += 1
        st.rerun()

    # --- CHAT HISTORY ---
    st.markdown("---")
    st.markdown("## 📝 Chat History")
    for entry in st.session_state.history:
        role = entry[0]
        msg  = entry[1]

        if role == "user":
            st.markdown(f"<div style='background-color:#E8F0FE;padding:8px;border-radius:8px;margin-bottom:4px;'><strong>🧑‍💻 You:</strong> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#F0F0F0;padding:8px;border-radius:8px;margin-bottom:4px;'><strong>🤖 Assistant:</strong> {msg['answer']}</div>", unsafe_allow_html=True)

            if len(entry) > 2:
                scores = entry[2]
                cols = st.columns(3)
                cols[0].metric("🎯 Relevance",    scores["relevance"])
                cols[1].metric("🧱 Groundedness", scores["groundedness"])
                cols[2].metric("🔧 Tool Use",     scores["tool_use"])


# -----------------------------
# VIDEO PLAYER
# -----------------------------
with col_video:
    st.subheader("Video Player")
    video_placeholder = st.empty()

    if st.session_state.current_embed:
        video_placeholder.markdown(
            f"""
            <iframe
                width="100%"
                height="420"
                src="{st.session_state.current_embed}&autoplay=1"
                frameborder="0"
                allow="autoplay; encrypted-media"
                allowfullscreen>
            </iframe>
            """,
            unsafe_allow_html=True
        )
    else:
        video_placeholder.markdown(
            """
            <div style="
                height:420px;
                border:2px dashed #aaa;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:18px;
                color:#777;
            ">
            🎬 Video will appear here
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -----------------------------
    # Relevant Video Segments
    # -----------------------------
    if st.session_state.history and st.session_state.history[-1][0] == "assistant":
        segments = st.session_state.history[-1][1]["segments"]

        if segments:
            st.markdown("### Relevant Video Segments")

            for i, seg in enumerate(segments):
                label = f"▶ {seg['start_time']}"
                if st.button(label, key=f"seg{i}"):
                    st.session_state.current_embed = seg["embed_link"]
                    st.rerun()
                st.caption(seg["watch_link"])
                st.caption(seg["text"][:120] + "...")
