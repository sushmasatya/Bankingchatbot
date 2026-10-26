import streamlit as st
import requests
import time

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Banking AI Assistant",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-online { background-color: #dcfce7; color: #166534; }
    .status-waking { background-color: #fef9c3; color: #854d0e; }
    .status-offline { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. DEFAULT CONFIGURATION
# ============================================================

DEFAULT_API_URL = "https://bankingchatbot-1.onrender.com"


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def check_backend_health(base_url: str):
    """Pings backend root endpoint to check connection and cold-start status."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/", timeout=6)
        if response.status_code == 200:
            return "online", "Connected"
        return "waking", f"Status: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return "waking", "Waking up from sleep (~30s)..."
    except requests.exceptions.ConnectionError:
        return "offline", "Cannot reach server"
    except Exception as e:
        return "offline", str(e)


def send_chat_query(base_url: str, question: str):
    """Sends question to FastAPI /chat endpoint."""
    url = f"{base_url.rstrip('/')}/chat"
    payload = {"question": question}
    try:
        response = requests.post(url, json=payload, timeout=75)
        if response.status_code == 200:
            data = response.json()
            return data.get("answer", "No answer received.")
        else:
            return f"⚠️ Backend error (HTTP {response.status_code}): {response.text}"
    except requests.exceptions.Timeout:
        return "⏳ Request timed out. Render's free tier was likely waking up. Please ask again in 10-15 seconds!"
    except requests.exceptions.ConnectionError:
        return "🔌 Could not connect to the backend. Please verify your Render URL or check if the service is live."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


# ============================================================
# 4. SIDEBAR SETTINGS & QUICK STARTERS
# ============================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_url = st.text_input(
        "Backend API URL",
        value=DEFAULT_API_URL,
        help="FastAPI server URL (e.g. deployed on Render or local http://localhost:8000)",
    )
    
    # Check backend connection
    status_type, status_text = check_backend_health(api_url)
    if status_type == "online":
        st.markdown(f'<span class="status-badge status-online">🟢 {status_text}</span>', unsafe_allow_html=True)
    elif status_type == "waking":
        st.markdown(f'<span class="status-badge status-waking">🟡 {status_text}</span>', unsafe_allow_html=True)
        st.caption("ℹ️ On Render's Free tier, the first request takes ~30-40s to boot the container.")
    else:
        st.markdown(f'<span class="status-badge status-offline">🔴 {status_text}</span>', unsafe_allow_html=True)
        st.caption("Ensure your Render service is deployed and status is 'Live'.")
        
    if st.button("🔄 Check Connection Now"):
        st.rerun()

    st.divider()

    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the daily ATM withdrawal limit?",
        "How much did I spend on food?",
        "What is my highest spending category?",
        "What was my total expenditure?",
        "How do I report a stolen debit card?",
        "What are the interest rates on personal loans?"
    ]

    selected_sample = None
    for q in sample_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            selected_sample = q

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Dual-Agent Architecture: RAG (Policy Docs) + Analytics (Pandas Expense Aggregation) powered by Google Gemini.")


# ============================================================
# 5. CHAT STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **Banking AI Assistant**.\n\n"
                "I can help you with:\n"
                "- 📜 **Banking Policies & Rules** (e.g. ATM limits, fees, card security, procedures)\n"
                "- 📊 **Expense Insights** (e.g. food spending, top categories, monthly breakdown)\n\n"
                "What would you like to know today?"
            ),
        }
    ]


# ============================================================
# 6. MAIN CHAT DISPLAY
# ============================================================

st.markdown('<div class="main-header">🏦 Banking AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about bank policies or analyze your recent expenditures</div>', unsafe_allow_html=True)

# Render all existing chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🏦"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input from either chat_input or sidebar sample buttons
user_query = st.chat_input("Type your banking or expense question here...")

prompt_to_process = selected_sample or user_query

if prompt_to_process:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    # 2. Call backend and display response
    with st.chat_message("assistant", avatar="🏦"):
        with st.spinner("Analyzing request and consulting banking agents..."):
            answer = send_chat_query(api_url, prompt_to_process)
            st.markdown(answer)

    # 3. Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
