import os
from dotenv import load_dotenv

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

# ============================================================
# 1. APPLICATION & MIDDLEWARE
# ============================================================

app = FastAPI(title="Banking AI Chatbot", version="1.0")

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
client = None
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Warning: Could not initialize Gemini client: {e}")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ============================================================
# 3. VECTOR DATABASE (RAG POLICY AGENT)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "banking_concepts_guide.pdf")

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
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
# 5. CORE HELPER FUNCTIONS
# ============================================================

def call_gemini(prompt: str) -> str:
    """Attempts generation via Gemini API with automatic model fallbacks."""
    if not client:
        return None

    models_to_try = [GEMINI_MODEL, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    for model_name in dict.fromkeys([m for m in models_to_try if m]):
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception as e:
            print(f"[Gemini] {model_name} failed: {e}")
            continue
    return None


def ask_rag(question: str) -> str:
    results = vector_db.similarity_search(question, k=3)
    retrieved_chunks = [doc.page_content for doc in results]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a banking customer support assistant.
Answer the customer's question using ONLY the provided context below.
If the answer is not in the context, state that you do not have enough information.

Context:
{context}

Customer Question:
{question}

Answer:"""

    ai_answer = call_gemini(prompt)
    if ai_answer:
        return ai_answer

    # Fallback directly to retrieved context
    if retrieved_chunks:
        return f"{retrieved_chunks[0]}\n\n*(Source: Banking Policy Guide)*"
    return "I do not have enough information in the policy documents to answer this question."


def ask_expense(question: str) -> str:
    q = question.lower()

    # Pre-calculated deterministic facts from Pandas
    cat_summary = ", ".join([f"{cat}: ₹{amt}" for cat, amt in category_expense.items()])

    prompt = f"""You are a banking financial assistant.
Answer the customer question using the expenditure data:
- Total expenditure: ₹{total_expense}
- Category breakdown: {cat_summary}
- Highest category: {highest_category} (₹{highest_amount})

Transactions:
{expenses[['date', 'description', 'category', 'expense_amount']].to_string(index=False)}

Customer question: {question}
Answer clearly and concisely."""

    ai_answer = call_gemini(prompt)
    if ai_answer:
        return ai_answer

    # Deterministic fallback logic directly from Pandas
    if "food" in q:
        food_items = expenses[expenses["category"] == "Food"]
        items_str = ", ".join([f"{row['description']}: ₹{row['expense_amount']}" for _, row in food_items.iterrows()])
        food_sum = int(food_items["expense_amount"].sum())
        return f"You spent **₹{food_sum:,}** on Food ({items_str})."
    
    if "shopping" in q:
        shop_sum = int(expenses[expenses["category"] == "Shopping"]["expense_amount"].sum())
        return f"You spent **₹{shop_sum:,}** on Shopping (Amazon)."

    if "highest" in q or "most" in q:
        return f"Your highest spending category is **{highest_category}** at **₹{highest_amount:,}**."

    if "total" in q or "overall" in q:
        return f"Your total expenditure is **₹{total_expense:,}** across {len(category_expense)} categories ({cat_summary})."

    return f"Your total expenditure is **₹{total_expense:,}**. Spending by category: {cat_summary}."


def chatbot(question: str) -> str:
    q = question.lower()
    expense_keywords = [
        "spent", "spend", "expenditure", "expense", "expenses",
        "spending", "food", "shopping", "bills", "travel",
        "entertainment", "reduce", "saving", "total", "highest", "salary"
    ]

    if any(k in q for k in expense_keywords):
        return ask_expense(question)
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
    return HTMLResponse("<h2>Banking Internal Assistant is running</h2>")

@app.get("/health")
def health():
    return {"status": "running", "chunks_indexed": len(chunks)}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = chatbot(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        return {"question": request.question, "answer": f"Unable to process question: {str(e)}"}
