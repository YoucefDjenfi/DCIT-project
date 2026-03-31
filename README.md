# DCIT Bot — RAG Cyber-Law Assistant

> This section documents the AI feature added to the Shellmates Discord Bot
> as part of the **Citoyenneté Numérique et IA** module project at ESI Alger.

---

## ⚖️ What it does

The bot adds a `/ask-law` command powered by **RAG (Retrieval-Augmented Generation)**.  
Students can ask questions in French about Algerian cyber law and digital citizenship.  
The bot retrieves the most relevant passages from official legal texts and uses an LLM
(Llama 3 via Groq) to formulate a cited, grounded answer — it does **not** hallucinate
laws that aren't in the documents.

---

## 🏗️ Architecture

```
knowledge_base/ (PDFs)
        │
        ▼
  ingest.py  ──── pypdf (text extraction)
                ── paraphrase-multilingual-MiniLM-L12-v2 (embeddings)
                ── ChromaDB (vector storage with priority metadata)
                         │
                         ▼
              rag_query.py  ──── tiered retrieval (P1 → P2 → P3 fallback)
                              ── Groq API / llama3-70b-8192
                                         │
                                         ▼
                      bot/cogs/cyber_law_ai.py  ──── /ask-law command
                                                   ── /law-help command
```

**Priority system:** documents are assigned a priority tier (1–3).
On every query, Priority 1 texts (core cyber laws) are searched first.
If similarity scores are insufficient, the search broadens to Priority 2,
then Priority 3. This ensures authoritative sources are always preferred.

---

## 📚 Knowledge Base — Legal Sources

| Document | Description | Language | Priority | Source |
|----------|-------------|----------|----------|--------|
| Loi n° 09-04 (2009) | Cybercriminalité | French | 1 | [CYRILLA](https://cyrilla.org) |
| Loi n° 18-07 (2018) | Protection des données personnelles | French | 1 | [UNIDIR](https://database.cyberpolicyportal.org) |
| Code pénal algérien (2016) | Articles 394 bis–394 nonies (TIC) | French | 1 | [CYRILLA](https://cyrilla.org) |
| Décret présidentiel 20-05 (2020) | Dispositif national SSI | French | 2 | Journal Officiel |
| Loi 20-06 (2020) | Modifications code pénal | French | 2 | Journal Officiel |
| Loi 15-04 (2015) | Signature et certification électroniques | French | 2 | Journal Officiel |
| Ordonnance 21-11 (2021) | Pôle pénal national TIC | French | 2 | Journal Officiel |
| Convention arabe sur la cybercriminalité (2010) | Traité international | FR (traduit) | 2 | Ligue arabe |
| Loi organique 12-05 (2012) | Loi sur l'information | French | 3 | Journal Officiel |
| Course materials (teacher PDFs) | Cours DCIT ESI | French | 1 | Provided by instructor |

---

## 🚀 Setup

### 1. Add required packages

```bash
pip install pypdf sentence-transformers chromadb groq
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```
DISCORD_TOKEN=your_token
GROQ_API_KEY=your_groq_key
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Add your PDFs

Place all legal text PDFs in the `knowledge_base/` folder.  
All documents must be in **French** (translate Arabic/English docs via DeepL first).  
Update `document_priorities.py` if you add new files.

### 4. Build the vector database

```bash
python ingest.py
```

This runs once and creates `chroma_db/`. Re-run only when you add new PDFs.

### 5. Run the bot

```bash
python main.py
```

---

## 💬 Commands

| Command | Description | Access |
|---------|-------------|--------|
| `/ask-law <question>` | Ask a question about Algerian cyber law | All users |
| `/law-help` | Show what the bot does and what laws it knows | All users |

### Example queries

- `/ask-law Quelles sont les sanctions pour accès frauduleux à un système informatique ?`
- `/ask-law Est-il légal de partager la photo de quelqu'un sans sa permission ?`
- `/ask-law Que dit la loi 18-07 sur la collecte de données personnelles ?`
- `/ask-law Quels sont les pouvoirs des autorités pour enquêter sur les cybercrimes ?`

---

## 📁 New Files Added

```
DCIT-project/
├── knowledge_base/          # Place your PDFs here (gitignored)
├── chroma_db/               # Auto-generated vector DB (gitignored)
├── ingest.py                # Run once to build the DB
├── rag_query.py             # RAG engine — imported by the cog
├── document_priorities.py   # Priority mapping for all PDFs
├── bot/cogs/cyber_law_ai.py # Discord Cog: /ask-law and /law-help
└── .env.example             # Environment variable template
```

---

## ⚠️ Limitations

- Answers are **indicative** and not a substitute for professional legal advice.
- The bot only knows what's in the documents — it will say so clearly if it can't find an answer.
- DB-dependent features (events, quiz, reminders) require a PostgreSQL connection
  (`DB_URL` in `.env`). The RAG feature works without it.

---

## 🙏 Acknowledgements

Base bot architecture: [sara-arz/shellmates-discord-bot](https://github.com/sara-arz/shellmates-discord-bot)  
Legal texts: Journal Officiel de la République Algérienne, CYRILLA Database, UNIDIR Cyber Policy Portal  
AI: [Groq](https://groq.com) / Meta Llama 3 70B  
Embeddings: [sentence-transformers](https://www.sbert.net) — `paraphrase-multilingual-MiniLM-L12-v2`
