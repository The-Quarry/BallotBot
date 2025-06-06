import streamlit as st
import requests
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import tempfile
import os
import uuid
import re

st.set_page_config(page_title="BallotBot - Guernsey Election 2025", layout="wide")

# --- Utility to create safe, unique keys ---
def make_safe_key(*parts):
    raw_key = "_".join(parts)
    return re.sub(r'\W+', '_', raw_key)

# --- State initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "liked_responses" not in st.session_state:
    st.session_state.liked_responses = []

if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --- Custom font styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    html, body, div, p, span, h1, h2, h3, h4, h5, h6, textarea {
        font-family: 'Poppins', sans-serif !important;
    }

    .stChatMessage, .stMarkdown, .stButton, .stTextInput {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Suggested prompt buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #f0f2f6 !important;
        border: 1px solid #d0d3d9 !important;
        border-radius: 8px !important;
        color: #333 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease-in-out;
        width: 100% !important;
        margin-bottom: 6px;
        text-align: center !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background-color: #e2e6eb !important;
        border-color: #c4c8cd !important;
    }

    /* Mobile adjustments */
    @media screen and (max-width: 768px) {
        .stButton > button {
            font-size: 16px !important;
            padding: 10px 14px !important;
        }

        .stChatMessage p, .stMarkdown p {
            font-size: 16px !important;
        }

        .stTextInput input {
            font-size: 16px !important;
        }

        .block-container {
            padding: 0rem 1rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Branding ---
st.sidebar.image("ballotbot_logo.png", use_container_width=True)
st.sidebar.title("🗳️ BallotBot")
st.sidebar.markdown("""
Overwhelmed by the mountain of campaign material in the **2025 Guernsey General Election**? We are here to help.

Ask what candidates say about:
- **GST**
- **Housing**
- **Air Links** and more...

BallotBot uses candidate manifestos, podcasts, hustings and more. They were built by The Quarry as part of our efforts to drive community engagement. AI has been used and all answers are regularly checked for accuracy. They will also update as candidates policy positions become clearer during the campaign.
""")

# --- Header ---
st.title("BallotBot - Election 2025")

API_URL = "https://ballotbot.onrender.com/chat"

# --- Suggested prompts ---
suggested_prompts = [
    "What do candidates say about housing?",
    "How do candidates view the economy?",
    "What are the views on education?",
    "What do candidates say about the health service?",
    "What are the views on taxation?",
    "Who supports GST?",
    "What do they think about government reform?",
    "What are the candidates’ ideas on transport?"
]

st.markdown("**Try a question below or type your own:**")
cols = st.columns(3)
for i, prompt in enumerate(suggested_prompts):
    if cols[i % 3].button(prompt, key=f"prompt_{i}"):
        st.session_state.pending_query = prompt

# --- Chat input ---
user_input = st.chat_input("Ask about a candidate or a policy...")
if user_input:
    st.session_state.pending_query = user_input

# --- Handle pending query only once ---
if st.session_state.pending_query:
    st.session_state.chat_history.append((st.session_state.pending_query, None))
    st.session_state.query = st.session_state.pending_query
    st.session_state.pending_query = None

# --- Handle response fetching if needed ---
if (
    st.session_state.chat_history and
    st.session_state.chat_history[-1][1] is None and
    not st.session_state.processing_query
):
    st.session_state.processing_query = True
    query = st.session_state.chat_history[-1][0]
    with st.spinner("Thinking..."):
        try:
            res = requests.post(API_URL, json={"query": query}, timeout=300)
            if res.status_code == 200:
                result = res.json().get("response", "No response received.")
            else:
                result = f"❌ Server error: {res.status_code}"
        except Exception as e:
            result = f"❌ Request failed: {e}"

        st.session_state.chat_history[-1] = (query, result)
    st.session_state.processing_query = False

# --- Display chat history ---
for query, result in st.session_state.chat_history:
    if result is not None:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("BallotBot", avatar="ballotbot_logo.png"):
            if isinstance(result, dict) and "primary" in result:
                st.markdown(result["primary"])
                with st.expander("💡 See what those opposed say"):
                    st.markdown(result["alternate"])
            elif isinstance(result, str):
                st.markdown(result)
            elif isinstance(result, list):
                for i, item in enumerate(result):
                    name = item.get("name", "Unknown")
                    text = item.get("summary") or item.get("statement") or "No content"
                    url = item.get("source_url", "")
                    name_md = f"[**{name}**]({url})" if url else f"**{name}**"
                    st.markdown(f"{name_md}: {text}")

                    unique_key = make_safe_key("save", query, name, str(i))
                    if st.button("✅ Save Candidate", key=unique_key):
                        if item not in st.session_state.liked_responses:
                            st.session_state.liked_responses.append(item)
                            st.success(f"{name} saved!")
            else:
                st.markdown("⚠️ Unrecognized response format.")

# --- Prepare PDF file if needed ---
temp_pdf_path = None
if st.session_state.get("liked_responses"):
    temp_pdf_path = os.path.join(tempfile.gettempdir(), "ballotbot_favourites.pdf")
    c = pdf_canvas.Canvas(temp_pdf_path, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "My Candidates created with BallotBot by The Quarry")
    y -= 30
    c.setFont("Helvetica", 12)

    for item in st.session_state["liked_responses"]:
        name = item.get("name", "Unknown")
        url = item.get("source_url", "")

        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)

        c.drawString(50, y, f"Candidate: {name}")
        y -= 20
        if url:
            c.drawString(50, y, f"URL: {url}")
            y -= 20

        y -= 10

    c.save()


# --- Main section: Saved Candidates + Download ---
with st.expander("✅ Your Saved Candidates", expanded=False):
    if st.session_state.liked_responses:
        for item in st.session_state.liked_responses:
            name = item.get("name", "Unknown")
            url = item.get("source_url", "")
            name_md = f"[**{name}**]({url})" if url else f"**{name}**"
            st.markdown(f"- {name_md}")
    else:
        st.markdown("_(You haven’t saved any candidates yet)_")

# --- Download PDF in main panel ---

if temp_pdf_path:
    with open(temp_pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download My Candidates (PDF)",
            data=pdf_file,
            file_name="ballotbot_favourites.pdf",
            mime="application/pdf",
            key="main_download"
        )
    

# --- Clear actions in sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 Manage Your Data")

# Button to clear chat history
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.query = ""
    st.session_state.pending_query = None
    st.session_state.processing_query = False
    st.rerun()

# Button to clear saved candidates
if st.sidebar.button("❌ Clear Saved Candidates"):
    st.session_state.liked_responses = []
    st.rerun()




