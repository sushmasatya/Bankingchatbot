import streamlit as st
import requests
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Internal Chat Assistant",
    page_icon="🏦",
    layout="centered"
)

# Header matching tutorial
st.title("Internal Chat Assistant")
st.caption("Ask anything about banking policies, products, or procedures.")

st.divider()
st.success("Ready — 120 document chunks indexed.", icon="✅")
st.divider()

# Backend API Configuration
DEFAULT_API_URL = os.getenv("BACKEND_URL", "https://bankingchatbot-1.onrender.com")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for msg in st.session_state.messages:
    avatar = "🧑‍🦰" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask a question...")

if user_input:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🦰"):
        st.markdown(user_input)

    # 2. Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                res = requests.post(
                    f"{DEFAULT_API_URL.rstrip('/')}/chat",
                    json={"question": user_input},
                    timeout=75
                )
                if res.status_code == 200:
                    answer = res.json().get("answer", "No answer received.")
                else:
                    answer = f"⚠️ Server error ({res.status_code}): {res.text}"
            except Exception as e:
                answer = f"⚠️ Connection error: {str(e)}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
