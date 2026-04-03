# 🤖 DCIT Discord Bot

> ⚠️ **This repository is retired and no longer maintained.**  
> It was my learning playground for Discord.py and bot architecture, forked from the Shellmates Discord bot.  
> The final, production‑ready bot is now a standalone project: **[DCIT-Bot](https://github.com/YoucefDjenfi/DCIT-Bot)** – a complete RAG‑based legal assistant for Algerian cyber law.  
> Please head there for the up‑to‑date code, documentation, and support.

A Discord bot built on top of the [Shellmates](https://github.com/sara-arz/shellmates-discord-bot) community bot, extended with an AI-powered legal assistant for ESI Alger's **Citoyenneté Numérique et IA** module.

---

## ✨ Features

### ⚖️ DCIT Cyber-Law Assistant *(new)*
An AI assistant that answers questions about Algerian cyber law and digital citizenship, grounded in official legal texts via RAG. It never invents laws — it only answers from the documents.

- `/ask-law <question>` — Ask anything about Algerian cyber law in French (or English)
- `/law-help` — Show what the bot does and which laws it knows

### 🛡️ Moderation
Banned word detection, role-based permissions, auto message filtering.

### 🎯 Events & Reminders
Create and manage community events with automatic DM reminders.

### ❓ Quiz System
Cybersecurity quiz with difficulty levels, points, and a leaderboard.

### 📚 Cyber Facts
Community-sourced cybersecurity facts database.

---

## ⚖️ How the RAG Assistant Works

```
knowledge_base/ (PDFs)
        │
        ▼
  ingest.py ──── pypdf (text extraction + boilerplate cleaning)
              ── article-aware chunking (splits on Art. X boundaries)
              ── paraphrase-multilingual-MiniLM-L12-v2 (embeddings)
              ── ChromaDB (vector DB with priority metadata)
                       │
                       ▼
            rag_query.py ──── query expansion (Nmap → "394 bis accès frauduleux", etc.)
                           ── cosine retrieval: top 40 candidates (wide net)
                           ── cross-encoder reranking: ms-marco-MiniLM-L-6-v2
                           ── priority boost: P1 docs score +0.5
                           ── Groq API / llama-3.3-70b-versatile
                                      │
                                      ▼
                  bot/cogs/cyber_law_ai.py ──── /ask-law
                                             ── /law-help
```

**Two-stage retrieval:** cosine similarity finds 40 candidate chunks quickly; a cross-encoder then re-scores each (question, chunk) pair semantically. This separates word overlap from actual meaning — ensuring `Nmap` maps to `Art. 394 bis` even though the words don't match.

**Query expansion:** technical terms (`Nmap`, `DDoS`, `phishing`, etc.) are mapped to French legal equivalents before embedding. Covers both French and English input.

**Priority boost:** Priority 1 documents (core cyber laws) receive a fixed score bonus in the reranker, ensuring they beat background documents when relevance is otherwise similar.

---

## 📚 Knowledge Base

| Document | Description | Priority | Language |
|----------|-------------|----------|----------|
| `DZ_FR_Cybercrime Law_2009.pdf` | Loi 09-04 — cybercriminalité | 1 | FR |
| `2016_Algeria_fr_Code Penal.pdf` | Code pénal complet — Art. 394 bis–nonies (TIC) | 1 | FR |
| `2018_Algeria_fr_Loi n_ 18-07...pdf` | Loi 18-07 — protection des données | 1 | FR |
| `Loi n° 18-07...pdf` | Loi 18-07 — copie cours ESI | 1 | FR |
| `Law 20-06 Algeria.pdf` | Modifications code pénal 2020 | 2 | FR |
| `2020_Algeria_fr_Décret présidentiel n_ 20-05...pdf` | Dispositif national SSI | 2 | FR |
| `Loi n∞ 15-04...pdf` | Signature et certification électroniques | 2 | FR |
| `Penal Procedure Code 2021 Update.pdf` | Pôle pénal TIC 2021 | 2 | FR |
| `Arab_Convention_Cybercrime_2010_FR.pdf.pdf` | Convention arabe cybercriminalité (FR) | 2 | FR |
| `Loi organique n° 12-05...pdf` | Loi sur l'information 2012 | 3 | FR |

> The full Penal Code (362 pages) is ingested. The cross-encoder handles relevance precision — only TIC-relevant articles make it into the final context.

---

## 🚀 Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YoucefDjenfi/DCIT-project.git
cd DCIT-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers --upgrade   # ensures CrossEncoder is available
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Fill in DISCORD_TOKEN and GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Add PDFs to `knowledge_base/`

Place all legal text PDFs in `knowledge_base/`. Documents should be in French where possible — translate non-French PDFs using [DeepL](https://www.deepl.com) before adding.

If you add a new file, add its exact filename to `document_priorities.py`.

### 4. Build the vector database

```bash
python ingest.py
```

This runs once and creates `chroma_db/`. Re-run whenever you add or update PDFs. The cross-encoder model (~80MB) is downloaded automatically on first bot startup.

### 5. Run the bot

```bash
python main.py
```

Slash commands sync automatically on startup. First run takes ~30–60 seconds while models load.

> **Note:** Events, quiz, and reminder features require a PostgreSQL database (`DB_URL` in `.env`). The RAG assistant works without it.

---

## 💬 Commands

### ⚖️ DCIT Legal Assistant

| Command | Description |
|---------|-------------|
| `/ask-law <question>` | Ask about Algerian cyber law (French or English) |
| `/law-help` | What the bot knows and how to use it |

**Example queries:**
```
/ask-law Quelles sont les sanctions pour accès frauduleux à un système informatique ?
/ask-law Est-il légal d'effectuer un scan Nmap sur un réseau Wi-Fi public ?
/ask-law Que dit la loi 18-07 sur la collecte de données personnelles ?
/ask-law Am I allowed to perform an Nmap scan on a public wifi I don't own?
/ask-law What are the penalties for unauthorized access to a computer system?
```

### 👥 Community Commands

| Command | Description |
|---------|-------------|
| `/quiz [difficulty]` | Cybersecurity quiz |
| `/leaderboard` | Quiz leaderboard |
| `/events` | Upcoming events |
| `/cyberfacts` | Browse cyber facts |

### 🛡️ Admin Commands

| Command | Description |
|---------|-------------|
| `/banword <word>` | Add banned word |
| `/add_event ...` | Create community event |
| `/addcyberfact <fact>` | Add a cyber fact |

---

## 📁 Project Structure

```
DCIT-project/
├── bot/
│   ├── cogs/
│   │   ├── cyber_law_ai.py       ← DCIT AI assistant
│   │   ├── cyberfacts_commands.py
│   │   ├── events_commands.py
│   │   ├── banned_words.py
│   │   ├── quiz_commands.py
│   │   └── ...
│   └── bot.py
├── database/
│   ├── Repositories/
│   └── connection.py             ← patched for lazy DB connection
├── knowledge_base/               ← PDFs go here (gitignored)
├── chroma_db/                    ← auto-generated vector DB (gitignored)
├── ingest.py                     ← run once to build the DB
├── rag_query.py                  ← RAG engine: expansion + cosine + reranker
├── document_priorities.py        ← PDF priority mapping (exact filenames)
├── config.py                     ← env vars including GROQ_API_KEY
├── main.py
└── .env.example
```

---

## ⚠️ Limitations

- Answers are **indicative** — not a substitute for professional legal advice.
- The bot only answers from its knowledge base. It says so clearly when it cannot find an answer.
- Events, quiz, and reminder features require a PostgreSQL connection.
- Non-French PDFs produce lower-quality retrieval. Translate to French before ingesting.

---

## 🙏 Acknowledgements

- Base bot: [sara-arz/shellmates-discord-bot](https://github.com/sara-arz/shellmates-discord-bot) — Shellmates Club, ESI Alger
- Legal texts: Journal Officiel de la République Algérienne, [CYRILLA](https://cyrilla.org), [UNIDIR](https://database.cyberpolicyportal.org)
- AI inference: [Groq](https://groq.com) — Meta Llama 3.3 70B Versatile
- Embeddings + reranking: [sentence-transformers](https://www.sbert.net) — `paraphrase-multilingual-MiniLM-L12-v2` + `ms-marco-MiniLM-L-6-v2`
