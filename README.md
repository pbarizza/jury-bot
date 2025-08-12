#  Jury-Bot: AI-Powered Startup Evaluation System

Jury-Bot is an intelligent, transparent, and scalable AI judge for startups. It evaluates, ranks, and answers questions about startups based on their data and pitch decks — all powered by **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **natural language reasoning**.

> 🔗 **Live Demo**: _Coming soon — link will be added here_

---

## What It Does

Jury-Bot helps investors, incubators, or competition organizers:
- **Search** 500+ startups by domain, region, team size, funding, and more
- **Understand** each startup deeply by analyzing its pitch deck and metadata
- **Rank** startups based on innovation, traction, team strength, and market fit
- **Answer complex queries** like:
  - "Which startups are from Saudi Arabia?"
  - "Show me AI startups with team size > 5"
  - "Why was SP-0042 ranked so high?"
- **Explain its decisions** with evidence from pitch decks and data

It’s not just a chatbot — it’s a **fair, consistent, and auditable AI juror**.

---

## Architecture Overview

Jury-Bot is built with a modular, extensible architecture:

```
+------------------+
|   Streamlit GUI   | ← Beautiful dark-mode chat interface
+------------------+
         ↓
+------------------+
|   FastAPI Backend | ← Orchestrates queries and responses
+------------------+
         ↓
+------------------+     +------------------+
|   Pinecone        |   |   Structured DB   |
| (Vector Search)   |   | (SQLite / Pandas) |
+------------------+   +------------------+
         ↓
+------------------+
|   LLM Reasoning   | ← Explains rankings (via Groq or local LLaMA)
+------------------+
```

### Core Components

| Component | Purpose |
|--------|---------|
| **Pinecone** | Stores semantic embeddings of startup data and pitch decks for deep retrieval |
| **FastAPI** | Backend API that handles queries and orchestrates responses |
| **Streamlit** | Interactive chat UI with dark mode and responsive design |
| **Sentence Transformers** | Generates 1024-dim embeddings (`BAAI/bge-large-en-v1.5`) |
| **LLM (Groq / Ollama)** | Powers reasoning and natural explanations (LLaMA 3) |
| **GitHub Secrets** | Securely stores API keys — never exposed in code |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/jury-bot.git
cd jury-bot
```

### 2. Set Up the RAG Backend

Ensure your FastAPI server is running:

```bash
cd api
pip install -r requirements.txt
python rag_startup_pipeline.py ingest   # Load data into Pinecone
python rag_startup_pipeline.py serve    # Start API
```

### 3. Run the Streamlit GUI

In a new terminal:

```bash
cd streamlit
pip install -r ../requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Security & Secrets

- API keys (Pinecone, Groq, etc.) are stored in `.env` during development
- In production, **GitHub Secrets** are used via Streamlit’s `st.secrets`
- The `.env` file is ignored in Git — no secrets are ever exposed

Example secret:
```env
BACKEND_URL=https://your-api.onrender.com/query
```

---

## Future Features (Roadmap)

- **Startup ranking engine** with explainable criteria
- **SQL-based filtering** (by country, domain, team size, revenue)
- **Session memory** to maintain conversation context
- **Audit logs** showing evidence for every decision
- **Export rankings** to PDF or CSV
- **MCP-inspired tool calling** for advanced reasoning

---

## Contributing

Contributions are welcome! Whether you're improving the UI, adding new query types, or enhancing the evaluation logic, feel free to open an issue or PR.

---

## License

MIT License — see `LICENSE` for details.

---

Built with ❤️ for fair, data-driven startup evaluation.
```

---

### Notes:
- Replace `your-username` in the clone URL.
- When your demo is live, update the **Live Demo** link.
- You can add badges later (e.g., `built-with-streamlit`, `pinecone-powered`) if desired.

This README is **professional, modular, and scalable** — perfect for sharing with teammates, judges, or investors.

Let me know when you want to add the **ranking engine** or **SQL filter API** — I’ll build it with you! 🚀
