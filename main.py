import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Banking AI Chatbot",
    description="Banking chatbot using FAISS, HuggingFace embeddings, Pandas and Gemini",
    version="1.0"
)


# ============================================================
# 2. GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# 3. LOAD BANKING POLICY PDF
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "banking_concepts_guide.pdf"
)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF file not found: {PDF_PATH}"
    )


loader = PyPDFLoader(PDF_PATH)

documents = loader.load()


# ============================================================
# 4. SPLIT PDF INTO CHUNKS
# ============================================================

text_splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=5
)

chunks = text_splitter.split_documents(documents)


# ============================================================
# 5. CREATE HUGGINGFACE EMBEDDINGS
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ============================================================
# 6. CREATE FAISS VECTOR DATABASE
# ============================================================

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)


# ============================================================
# 7. TRANSACTION DATA
# ============================================================

data = {
    "date": [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
        "2026-08-07",
        "2026-08-08"
    ],

    "description": [
        "Salary",
        "Swiggy",
        "Amazon",
        "Uber",
        "Electricity Bill",
        "Grocery",
        "Netflix",
        "Restaurant"
    ],

    "amount": [
        50000,
        -500,
        -3500,
        -300,
        -1500,
        -2500,
        -649,
        -1200
    ],

    "category": [
        "Income",
        "Food",
        "Shopping",
        "Travel",
        "Bills",
        "Food",
        "Entertainment",
        "Food"
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 8. PREPARE EXPENSE DATA
# ============================================================

expenses = df[df["amount"] < 0].copy()

expenses["expense_amount"] = expenses["amount"].abs()


# ============================================================
# 9. RETRIEVE INFORMATION FROM FAISS
# ============================================================

def retrieve(question, n_results=3):

    results = vector_db.similarity_search(
        question,
        k=n_results
    )

    retrieved_chunks = []
    sources = []

    for document in results:

        retrieved_chunks.append(
            document.page_content
        )

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        sources.append(source)

    return retrieved_chunks, sources


# ============================================================
# 10. RAG CHATBOT
# ============================================================

def ask_rag(question, n_results=3):

    retrieved_chunks, sources = retrieve(
        question,
        n_results
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a banking customer support assistant.

Answer the customer's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say that you do not have enough information.

Do not make up information.

Context:
{context}

Customer Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# 11. EXPENSE CHATBOT
# ============================================================

def ask_expense(question):

    total_expense = expenses[
        "expense_amount"
    ].sum()

    category_expense = (
        expenses
        .groupby("category")["expense_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_category = category_expense.index[0]

    highest_amount = category_expense.iloc[0]

    category_details = "\n".join(
        [
            f"{category}: ₹{amount}"
            for category, amount
            in category_expense.items()
        ]
    )

    prompt = f"""
You are a banking financial assistant.

Answer the customer's question using the customer's
transaction information provided below.

Total expenditure:
₹{total_expense}

Expenditure by category:
{category_details}

Highest spending category:
{highest_category}

Highest spending amount:
₹{highest_amount}

Customer question:
{question}

Give a simple and clear answer.

Do not invent transaction information.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# 12. CHATBOT ROUTER
# ============================================================

def chatbot(question):

    question_lower = question.lower()

    expense_keywords = [
        "spent",
        "spend",
        "expenditure",
        "expense",
        "expenses",
        "spending",
        "food",
        "shopping",
        "bills",
        "travel",
        "entertainment",
        "reduce",
        "saving"
    ]

    if any(
        keyword in question_lower
        for keyword in expense_keywords
    ):

        return ask_expense(question)

    else:

        return ask_rag(question)


# ============================================================
# 13. REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str


# ============================================================
# 14. HOME API
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Banking AI Chatbot is running"
    }


# ============================================================
# 15. CHAT API
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    answer = chatbot(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }