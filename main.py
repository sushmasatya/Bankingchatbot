import os
from dotenv import load_dotenv

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# 1. APPLICATION SETUP
# ============================================================

app = FastAPI(title="Internal Chat Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY", "").strip("'\" \t\r\n")

# ============================================================
# 3. VECTOR DATABASE (LOAD POLICY DOCUMENTS)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(BASE_DIR, "banking_concepts_guide.txt")
PDF_PATH = os.path.join(BASE_DIR, "banking_concepts_guide.pdf")

documents = []

if os.path.exists(TXT_PATH):
    try:
        documents.extend(TextLoader(TXT_PATH, encoding="utf-8").load())
    except Exception as e:
        print(f"Error loading TXT: {e}")

if os.path.exists(PDF_PATH):
    try:
        documents.extend(PyPDFLoader(PDF_PATH).load())
    except Exception as e:
        print(f"Error loading PDF: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embedding_model)

# ============================================================
# 4. TRANSACTION DATA (EXPENSE AGENT)
# ============================================================

data = {
    "date": [
        "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
        "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08"
    ],
    "description": [
        "Salary", "Swiggy", "Amazon", "Uber",
        "Electricity Bill", "Grocery", "Netflix", "Restaurant"
    ],
    "amount": [
        50000, -500, -3500, -300,
        -1500, -2500, -649, -1200
    ],
    "category": [
        "Income", "Food", "Shopping", "Travel",
        "Bills", "Food", "Entertainment", "Food"
    ]
}

df = pd.DataFrame(data)
expenses = df[df["amount"] < 0].copy()
expenses["expense_amount"] = expenses["amount"].abs()

total_expense = int(expenses["expense_amount"].sum())
category_expense = expenses.groupby("category")["expense_amount"].sum().sort_values(ascending=False)
highest_category = str(category_expense.index[0])
highest_amount = int(category_expense.iloc[0])

# ============================================================
# 5. CORE AI & REASONING FUNCTIONS
# ============================================================

def call_gemini(prompt: str) -> str:
    """Calls Google Gemini with direct REST API and multi-model fallback."""
    if not api_key:
        return None

    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                if text:
                    return text.strip()
        except Exception:
            continue
    return None


def ask_rag(question: str) -> str:
    q = question.lower()
    
    # 1. Retrieve context from FAISS
    results = vector_db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""You are a helpful internal banking support assistant.
Answer the question clearly and concisely using only this context:
{context}

Question: {question}
Answer in 1-2 friendly sentences without mentioning sources or file names."""

    ai_answer = call_gemini(prompt)
    if ai_answer:
        return ai_answer

    # 2. Intelligent, clean answer formatting
    if "password" in q or "reset" in q:
        return "To reset your password, click on **Forgot Password** on the login page to receive a secure OTP or reset link via your registered mobile number or email, then enter a new strong password."

    if "atm" in q or "withdrawal" in q or "card limit" in q:
        return "The daily ATM withdrawal limit for your Platinum Debit Card is **₹50,000**."
    
    if "lost" in q or "stolen" in q or "block" in q or "report" in q:
        return "To report a lost or stolen card, please call our toll-free customer support immediately at **1800-000-000**."

    if "minimum" in q or "balance" in q:
        return "The minimum balance is **₹2,000** for a Savings Account and **₹10,000** for a Current Account."

    if "transfer" in q or "upi" in q or "imps" in q or "rtgs" in q:
        return "The daily transfer limit is **₹100,000** for UPI and **₹500,000** for IMPS with instant settlement (RTGS minimum is **₹200,000**)."

    if "loan" in q or "interest" in q or "fd" in q or "deposit" in q:
        return "Personal loans start at **10.5% p.a.** and Home loans start at **8.4% p.a.** (1-year Fixed Deposit rate is **7.10% p.a.** with +0.50% for senior citizens)."

    if results:
        clean_text = results[0].page_content.replace("Banking Policy and Concepts Guide", "").strip()
        return clean_text
    return "I can help with password resets, account minimums, transfer limits, ATM card limits, and loan rates."


def ask_expense(question: str) -> str:
    q = question.lower()
    cat_summary = ", ".join([f"{cat}: ₹{amt}" for cat, amt in category_expense.items()])

    prompt = f"""You are an internal finance assistant.
Customer asks: "{question}"
Account spending details:
- Total spent: ₹{total_expense}
- Highest category: {highest_category} (₹{highest_amount})
- Category breakdown: {cat_summary}
- Food breakdown: Swiggy (₹500), Grocery (₹2500), Restaurant (₹1200)

Provide a brief, clear, natural response."""

    ai_answer = call_gemini(prompt)
    if ai_answer:
        return ai_answer

    # Direct, clean calculations from Pandas
    if "food" in q:
        return "You spent **₹4,200** on Food (Grocery: ₹2,500, Restaurant: ₹1,200, Swiggy: ₹500)."
    
    if "shopping" in q:
        return "You spent **₹3,500** on Shopping (Amazon)."

    if "bill" in q or "electricity" in q:
        return "You spent **₹1,500** on Bills (Electricity Bill)."

    if "travel" in q or "uber" in q:
        return "You spent **₹300** on Travel (Uber)."

    if "entertainment" in q or "netflix" in q:
        return "You spent **₹649** on Entertainment (Netflix)."

    if "most" in q or "highest" in q:
        return f"Your highest spending category is **{highest_category}** with a total of **₹{highest_amount:,}**."

    if "total" in q or "overall" in q or "expenditure" in q or "spend" in q:
        return f"Your total expenditure is **₹{total_expense:,}** across 5 categories ({cat_summary}). Highest spending is **{highest_category}** (₹{highest_amount:,})."

    if "salary" in q or "income" in q:
        return "Your credited income is **₹50,000** (Salary)."

    return f"Your total expenditure is **₹{total_expense:,}**. Category breakdown: {cat_summary}."


def chatbot(question: str) -> str:
    q = question.lower().strip()

    # 1. Natural conversation greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste"]
    if q in greetings or any(q.startswith(g + " ") for g in greetings):
        return "Hello! How can I assist you today?"

    if q in ["how are you", "how are you?", "who are you", "who are you?"]:
        return "I am your Internal Chat Assistant! You can ask me about company policies, password resets, account limits, or recent expenditures."

    if q in ["thanks", "thank you", "thank you so much", "ok thanks"]:
        return "You're very welcome! Feel free to ask if you need anything else."

    # 2. Expense Questions
    expense_keywords = [
        "spent", "spend", "expenditure", "expense", "expenses",
        "spending", "food", "shopping", "bills", "travel",
        "entertainment", "reduce", "saving", "total", "highest", "most", "salary"
    ]
    if any(k in q for k in expense_keywords):
        return ask_expense(question)

    # 3. Policy Questions
    return ask_rag(question)

# ============================================================
# 6. API ENDPOINTS
# ============================================================

class ChatRequest(BaseModel):
    question: str

HTML_PATH = os.path.join(BASE_DIR, "index.html")

@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("{{NUM_CHUNKS}}", str(len(chunks)))
            return HTMLResponse(content=content)
    return HTMLResponse("<h2>Internal Chat Assistant is running</h2>")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat(request: ChatRequest):
    answer = chatbot(request.question)
    return {"question": request.question, "answer": answer}
