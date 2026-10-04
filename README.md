# Banking AI Chatbot POC

A dual-agent Banking AI Assistant combining:
1. **Policy Agent (RAG)**: LangChain + HuggingFace Embeddings (`all-MiniLM-L6-v2`) + FAISS Vector Store + Gemini API.
2. **Expense Agent (Data Analytics)**: Pandas transaction aggregation + Gemini API.
3. **Query Router**: Automatic classification of incoming customer inquiries.

---

## Architecture Flow

```text
                  User
                   ↓
         Banking Chatbot API
              (FastAPI)
                   ↓
              Query Router
              ↙          ↘
      Policy question    Expense question
             ↓                  ↓
        FAISS + RAG          Pandas
             ↓                  ↓
         Gemini API ←───────────┘
             ↓
         Final Answer
```

---

## Project Structure

```text
banking-chatbot/
├── main.py              # Production-ready FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Optimized container build (CPU-only PyTorch & pre-cached model)
├── docker-compose.yml   # Local testing container config
├── .dockerignore        # Excludes unnecessary files from image
├── .env.example         # Example environment variables
└── README.md            # Cloud deployment instructions
```

---

## Cloud Deployment Guide

### Option 1: Google Cloud Run (Recommended)
Google Cloud Run is serverless, scales down to zero when idle (saving costs), and provides 2 million free requests per month.

#### Prerequisites:
1. Install [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk).
2. Authenticate: `gcloud auth login`.

#### Deploy in one command:
```bash
# 1. Set your project
gcloud config set project YOUR_PROJECT_ID

# 2. Deploy directly from source (Cloud Build handles the Docker build)
gcloud run deploy banking-chatbot \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="your_api_key_here",GEMINI_MODEL="gemini-2.5-flash" \
  --memory 2Gi \
  --cpu 1
```

---

### Option 2: Render.com (Easiest & Free/Low Cost)
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/banking-chatbot.git
   git push -u origin main
   ```
2. Go to [render.com](https://render.com) and log in.
3. Click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. In **Environment**, select **Docker**.
6. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: `your_actual_gemini_api_key`
   - `GEMINI_MODEL`: `gemini-2.5-flash`
7. Click **Create Web Service**. Render builds and hosts it at `https://your-app.onrender.com`.

---

### Option 3: Railway.app (Instant Setup)
1. Go to [railway.app](https://railway.app) and sign in.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your repository. Railway automatically detects the `Dockerfile`.
4. Go to **Variables** tab and add:
   - `GEMINI_API_KEY`: `your_actual_gemini_api_key`
5. Go to **Settings** → **Networking** → Click **Generate Domain**.
6. Done! Your API is live.

---

### Option 4: Hugging Face Spaces (Free CPU Docker)
Hugging Face offers free CPU hosting (2 vCPU, 16 GB RAM):
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Select **Docker** (Blank) SDK.
3. In Space Settings, add secret: `GEMINI_API_KEY`.
4. Clone the space repo and push these project files into it.

---

## Local Testing

### Using Docker Compose
```bash
# Copy and update .env
cp .env.example .env

# Run container
docker compose up --build
```

### Direct Python Execution
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"  # Windows PowerShell: $env:GEMINI_API_KEY="your_api_key_here"
uvicorn main:app --reload --port 8000
```

### Testing Endpoints

**Health Check:**
```bash
curl http://localhost:8000/
```

**Ask Expense Question:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How much did I spend on food?"}'
```

**Ask Policy Question:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the daily ATM withdrawal limit?"}'
```
