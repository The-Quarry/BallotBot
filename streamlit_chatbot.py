import streamlit as st
import requests

st.set_page_config(page_title="Guernsey Election 2025", layout="wide")
st.title("Guernsey Election 2025: BallotBot is here to help")

API_URL = "https://ballotbot.onrender.com"

# --- Reset Chat Button ---
if st.button("🔄 Reset Chat"):
    st.session_state.clear()
    st.rerun()

# --- Suggested prompts ---
suggested_prompts = [
    "What do candidates say about housing?",
    "How do candidates view the economy?",
    "What are the views on education?",
    "What do candidates say about the health service?",
    "What are the views on taxation?",
    "Who supports GST?",
    "What’s being proposed on government reform?",
    "What are the candidates’ ideas on transport?"
]

st.markdown("**Try one of these questions. You can also ask about a specific candidate's view on a topic. Or try your own (just don't be rude!).**")
cols = st.columns(3)
for i, prompt in enumerate(suggested_prompts):
    if cols[i % 3].button(prompt):
        st.session_state.query = prompt

# --- Initialize state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "visible_counts" not in st.session_state:
    st.session_state.visible_counts = {}

# --- Chat Input ---
query = st.text_input("Ask a question about a candidate or a policy:", key="query")

# --- Call API only if query is new ---
if query and (st.session_state.get("last_query") != query):
    with st.spinner("Thinking..."):
        try:
            res = requests.post(API_URL, json={"query": query}, timeout=300)
            if res.status_code == 200:
                result = res.json().get("response", "No response received.")
                st.session_state.chat_history.insert(0, (query, result))
                st.session_state.last_query = query
                st.session_state.visible_counts[query] = 10  # show first 10
            else:
                error_msg = f"❌ Server error: {res.status_code}"
                st.session_state.chat_history.insert(0, (query, error_msg))
        except Exception as e:
            error_msg = f"❌ Request failed: {e}"
            st.session_state.chat_history.insert(0, (query, error_msg))

# --- Display chat history ---

st.markdown("---")
st.subheader("Chat History")

for idx, (user_q, bot_r) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {user_q}")

    # If response is a dictionary (e.g. from stance cache)
    if isinstance(bot_r, dict):
        if "primary" in bot_r and "alternate" in bot_r:
            st.markdown("**BallotBot:**")
            st.markdown(bot_r["primary"])

            with st.expander("💡 See what those opposed say"):
                st.markdown(bot_r["alternate"])
        else:
            st.markdown("**BallotBot:** Unrecognized structured response.")
    
    # If it's a string (e.g. general fallback or candidate answer)
    elif isinstance(bot_r, str):
        st.markdown(f"**BallotBot:** {bot_r}")

    # If it's a list (e.g. list of candidate summaries)
    elif isinstance(bot_r, list) and all(isinstance(item, dict) for item in bot_r):
        key_prefix = f"response_{idx}"
        visible_key = f"{key_prefix}_visible"
        if visible_key not in st.session_state:
            st.session_state[visible_key] = 10

        visible_count = st.session_state[visible_key]
        for item in bot_r[:visible_count]:
            name = item.get("name", "Unknown")
            text = item.get("summary") or item.get("statement") or "No content"
            st.markdown(f"**{name}**: {text}")

        if visible_count < len(bot_r):
            if st.button("Show More", key=f"{key_prefix}_btn"):
                st.session_state[visible_key] += 10
                st.rerun()

    else:
        st.markdown("**BallotBot:** ⚠️ Unrecognized response format.")

    st.markdown("---")
