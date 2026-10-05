import os
import logging
from contextlib import asynccontextmanager
from typing import List, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banking-chatbot")

# ============================================================
# 1. CONFIGURATION & GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "banking_concepts_guide.pdf")
TXT_PATH = os.path.join(BASE_DIR, "banking_concepts_guide.txt")

# ============================================================
# 2. VECTOR DATABASE INITIALIZATION (FAISS + HuggingFace)
# ============================================================

vector_db = None

DEFAULT_POLICY_TEXT = """
COMPREHENSIVE REFERENCE GUIDE - Core Banking & Personal Finance Fundamentals

01 Savings Account
A savings account serves as the foundational cornerstone of personal financial management, designed primarily to provide a secure, liquid repository for your surplus funds while earning a modest interest return. Unlike checking accounts that are optimized for high-frequency daily transactions, savings accounts are engineered to help individuals build emergency reserves and preserve capital over the medium to long term. Funds held in a regulated savings account are typically protected by statutory deposit insurance schemes up to a specified limit, guaranteeing a high level of principal security. Accrued interest is periodically credited—often monthly or quarterly—based on daily or average monthly balances.

02 Credit Cards
A credit card operates as a revolving line of unsecured consumer credit issued by a financial institution, allowing cardholders to purchase goods, services, or access short-term funds against a pre-approved maximum credit limit. When you make a purchase using a credit card, the issuing bank covers the payment upfront, creating a short-term balance that you are obligated to repay during the subsequent billing cycle. If the statement balance is settled in full by the due date, cardholders effectively benefit from an interest-free grace period ranging from 20 to 50 days. Carrying an unpaid balance triggers high annual percentage rates (APR). Beyond purchasing power, credit cards offer fraud protection, zero-liability policies on unauthorized transactions, reward points, and cashback.

03 Loans
A loan is a formal financial agreement wherein a lender disburses a defined lump sum of money to a borrower with the explicit condition that the principal amount be repaid over an agreed schedule, accompanied by predetermined interest and service fees. Loans are broadly categorized into secured loans (such as home mortgages or auto financing, requiring tangible collateral) and unsecured loans (such as personal loans or educational debt, relying solely on borrower creditworthiness). Mechanics of repayment typically follow an amortization schedule of equal monthly installments (EMI).

04 Transfer Limits
Transfer limits are predefined daily, weekly, or transactional caps enforced by banking networks and financial institutions to regulate the volume and velocity of money moving across accounts. These limits minimize systemic risk and safeguard consumer funds against fraud, unauthorized account takeovers, or cyberattacks. Banks establish tiered thresholds across domestic clearinghouses, real-time instant payment interfaces, wire systems, or international wire networks. Account holders can often customize daily operational limits within their digital banking security consoles.

05 Password Reset
The password reset process represents a critical identity verification and account recovery protocol designed to restore authorized user access while maintaining strict defenses against unauthorized intrusion. When requested, the financial platform dispatches a time-sensitive, single-use reset link or a cryptographic one-time password (OTP) to the registered email address or verified mobile device, often supplemented by biometric scans or security questions.

06 ATM Withdrawals
Automated Teller Machine (ATM) withdrawals provide account holders with instant, physical cash access around the clock. The cardholder inserts an EMV chip debit card and inputs a private PIN. To prevent catastrophic balance depletion in cases of lost or cloned cards, financial institutions enforce strict per-transaction and daily maximum withdrawal limits on physical cash dispatches. Additionally, using an out-of-network ATM may incur convenience surcharges.

07 Account Charges
Account charges encompass the full schedule of standard service fees, administrative costs, and penalty tariffs levied by a banking institution. Common recurring charges include monthly maintenance fees, non-sufficient funds (NSF) penalties, overdraft surcharges, paper statement delivery fees, inactivity costs on dormant accounts, and foreign transaction markups. In many cases, fees can be waived by maintaining a minimum average daily balance or setting up direct payroll deposits.
"""

def init_vector_store():
    global vector_db
    logger.info("Initializing HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    if os.path.exists(PDF_PATH):
        try:
            logger.info(f"Loading policy document from {PDF_PATH}...")
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(chunks)} chunks from PDF.")
        except Exception as e:
            logger.warning(f"Error reading PDF ({e}), falling back to text.")

    if not chunks and os.path.exists(TXT_PATH):
        logger.info(f"Loading policy document from {TXT_PATH}...")
        loader = TextLoader(TXT_PATH, encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.info("Using embedded default policy knowledge guide.")
        chunks = text_splitter.create_documents([DEFAULT_POLICY_TEXT])

    vector_db = FAISS.from_documents(chunks, embedding_model)
    logger.info(f"FAISS vector store successfully created with {len(chunks)} chunks.")

# ============================================================
# 3. TRANSACTION DATA (PANDAS)
# ============================================================

columns = ["date", "description", "amount", "category"]
data = [
    ["2026-08-01", "Salary", 50000, "Income"],
    ["2026-08-02", "Swiggy", -500, "Food"],
    ["2026-08-03", "Amazon", -3500, "Shopping"],
    ["2026-08-04", "Uber", -300, "Travel"],
    ["2026-08-05", "Electricity Bill", -1500, "Bills"],
    ["2026-08-06", "Grocery", -2500, "Food"],
    ["2026-08-07", "Netflix", -649, "Entertainment"],
    ["2026-08-08", "Restaurant", -1200, "Food"]
]

df = pd.DataFrame(data, columns=columns)
expenses = df[df["amount"] < 0].copy()
expenses["expense_amount"] = expenses["amount"].abs()

# ============================================================
# 4. LIFESPAN & FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vector_store()
    yield

app = FastAPI(
    title="Banking AI Chatbot",
    description="Banking chatbot with FAISS, HuggingFace embeddings, Pandas, and Gemini API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 5. RETRIEVAL & RAG BOT
# ============================================================

def retrieve(question: str, n_results: int = 3) -> Tuple[List[str], List[dict]]:
    if vector_db is None:
        return ["Vector database not initialized."], [{"source": "System"}]

    results = vector_db.similarity_search(question, k=n_results)
    chunks = [doc.page_content for doc in results]
    sources = [doc.metadata if doc.metadata else {"source": "banking_concepts_guide.pdf"} for doc in results]
    return chunks, sources


def ask_rag(question: str, n_results: int = 3) -> str:
    if not client:
        return "GEMINI_API_KEY is not configured. Please set the environment variable."

    chunks, sources = retrieve(question, n_results)
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful banking customer support assistant.

Answer the question using ONLY the provided context below.
If the context does not contain enough information to answer the question, say:
"I don't have enough information to answer this question."

Do not make up information.
Keep the answer concise and clear.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini in ask_rag: {e}")
        return f"AI Service error: {str(e)}"

# ============================================================
# 6. EXPENSE BOT
# ============================================================

def ask_expense(question: str) -> str:
    if not client:
        return "GEMINI_API_KEY is not configured. Please set the environment variable."

    total_expense = expenses["expense_amount"].sum()
    category_expense = (
        expenses.groupby("category")["expense_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_category = category_expense.index[0] if not category_expense.empty else "None"
    highest_amount = category_expense.iloc[0] if not category_expense.empty else 0

    category_details = "\n".join(
        [f"{category}: ₹{amount}" for category, amount in category_expense.items()]
    )

    prompt = f"""You are a banking financial assistant.

Use ONLY the calculated transaction information below.

Total expenditure: ₹{total_expense}

Category-wise expenditure:
{category_details}

Highest spending category: {highest_category}
Highest spending amount: ₹{highest_amount}

Customer question:
{question}

Answer clearly and concisely.
Do not invent or change any numbers."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini in ask_expense: {e}")
        return f"AI Service error: {str(e)}"

# ============================================================
# 7. CHATBOT ROUTER
# ============================================================

def chatbot(question: str) -> Tuple[str, str]:
    question_lower = question.lower()
    expense_keywords = [
        "spent", "spend", "expenditure", "expense", "expenses",
        "spending", "food", "shopping", "bills", "travel",
        "entertainment", "reduce", "saving", "salary", "balance", "budget"
    ]

    if any(keyword in question_lower for keyword in expense_keywords):
        return ask_expense(question), "expense_agent"
    else:
        return ask_rag(question), "policy_rag_agent"

# ============================================================
# 8. REQUEST & RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(..., example="What is a credit card grace period?")

class ChatResponse(BaseModel):
    question: str
    routed_to: str
    answer: str

# ============================================================
# 9. API ENDPOINTS
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Banking AI Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }
        body { background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; }
        .header { background: #1e293b; border-bottom: 1px solid #334155; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
        .header-title { display: flex; align-items: center; gap: 12px; }
        .header-title h1 { font-size: 1.15rem; font-weight: 600; color: #fff; }
        .status-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; background: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 9999px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; max-width: 860px; width: 100%; margin: 0 auto; }
        .msg { display: flex; flex-direction: column; max-width: 80%; animation: fadeIn 0.25s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .msg.user { align-self: flex-end; }
        .msg.bot { align-self: flex-start; }
        .bubble { padding: 12px 18px; border-radius: 16px; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user .bubble { background: #2563eb; color: #ffffff; border-bottom-right-radius: 4px; }
        .msg.bot .bubble { background: #1e293b; color: #f1f5f9; border: 1px solid #334155; border-bottom-left-radius: 4px; }
        .badge { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; padding: 2px 8px; border-radius: 6px; display: inline-block; width: fit-content; }
        .badge.expense { background: #3730a3; color: #a5b4fc; }
        .badge.policy { background: #065f46; color: #6ee7b7; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
        .chip { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 8px 14px; border-radius: 9999px; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; }
        .chip:hover { background: #334155; color: #fff; border-color: #64748b; }
        .input-bar { background: #1e293b; border-top: 1px solid #334155; padding: 16px 24px; }
        .input-form { max-width: 860px; margin: 0 auto; display: flex; gap: 12px; }
        .input-form input { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 12px 16px; color: #fff; font-size: 0.95rem; outline: none; }
        .input-form input:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
        .input-form button { background: #2563eb; color: #fff; border: none; border-radius: 12px; padding: 0 22px; font-weight: 600; font-size: 0.95rem; cursor: pointer; transition: background 0.2s; }
        .input-form button:hover { background: #1d4ed8; }
        .input-form button:disabled { background: #475569; cursor: not-allowed; }
        .links { text-align: center; margin-top: 8px; font-size: 0.78rem; color: #64748b; }
        .links a { color: #3b82f6; text-decoration: none; }
        .typing { display: inline-flex; gap: 4px; align-items: center; padding: 8px 14px; }
        .typing span { width: 6px; height: 6px; background: #94a3b8; border-radius: 50%; animation: bounce 1.2s infinite ease-in-out; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <span style="font-size: 1.5rem;">🏦</span>
            <div>
                <h1>Banking AI Assistant</h1>
                <p style="font-size: 0.75rem; color: #94a3b8;">Dual-Agent: Policy RAG & Expense Analytics</p>
            </div>
        </div>
        <div class="status-badge"><span class="status-dot"></span> Live on Cloud</div>
    </div>

    <div class="chat-container" id="chat">
        <div class="msg bot">
            <div class="bubble">
                👋 <strong>Hello! I am your Banking AI Assistant.</strong><br><br>
                I can assist you with:
                <ul style="margin: 8px 0 8px 20px;">
                    <li><strong>Personal Expenses:</strong> Check food spend, salary, highest category, total expenditure.</li>
                    <li><strong>Banking Policies:</strong> Rules on savings accounts, credit cards, loans, ATM limits, transfer limits, and fees.</li>
                </ul>
                Try tapping one of the common questions below:
                <div class="suggestions">
                    <button class="chip" onclick="ask('How much did I spend on food?')">🍕 Food Spending</button>
                    <button class="chip" onclick="ask('What was my highest expense category?')">📊 Highest Expense</button>
                    <button class="chip" onclick="ask('What is the daily ATM withdrawal limit?')">🏧 ATM Limit</button>
                    <button class="chip" onclick="ask('What is the interest-free grace period for credit cards?')">💳 Credit Card Grace Period</button>
                </div>
            </div>
        </div>
    </div>

    <div class="input-bar">
        <form class="input-form" onsubmit="event.preventDefault(); send();">
            <input type="text" id="userInput" placeholder="Ask a question about policies or your spending..." autocomplete="off">
            <button type="submit" id="sendBtn">Send</button>
        </form>
        <div class="links">
            API Documentation: <a href="/docs" target="_blank">Swagger /docs</a> | Health: <a href="/health" target="_blank">/health</a>
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function ask(text) {
            input.value = text;
            send();
        }

        async function send() {
            const question = input.value.trim();
            if (!question) return;

            addMessage(escapeHtml(question), 'user');
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

            const typingId = addTyping();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                removeTyping(typingId);

                const badge = data.routed_to === 'expense_agent' 
                    ? '<span class="badge expense">📊 Personal Expense Agent</span>' 
                    : '<span class="badge policy">📘 Policy Guide (RAG)</span>';

                addMessage(badge + '<br>' + formatAnswer(data.answer), 'bot');
            } catch (err) {
                removeTyping(typingId);
                addMessage('⚠️ Error communicating with server. Please try again.', 'bot');
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
            }
        }

        function addMessage(html, type) {
            const div = document.createElement('div');
            div.className = 'msg ' + type;
            div.innerHTML = '<div class="bubble">' + html + '</div>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addTyping() {
            const id = 'typing-' + Date.now();
            const div = document.createElement('div');
            div.className = 'msg bot';
            div.id = id;
            div.innerHTML = '<div class="bubble typing"><span></span><span></span><span></span></div>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return id;
        }

        function removeTyping(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        function formatAnswer(text) {
            return escapeHtml(text).replace(/\\n/g, '<br>');
        }

        function escapeHtml(text) {
            const p = document.createElement('p');
            p.textContent = text;
            return p.innerHTML;
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=HTML_PAGE)

@app.get("/status")
def status():
    return {
        "status": "running",
        "service": "Banking AI Chatbot",
        "model": GEMINI_MODEL,
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, routed_to = chatbot(request.question)
    return ChatResponse(
        question=request.question,
        routed_to=routed_to,
        answer=answer
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
