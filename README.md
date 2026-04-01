# 🤖 DCIT Discord Bot

A Discord bot built on top of the [Shellmates](https://github.com/sara-arz/shellmates-discord-bot) community bot, extended with an AI-powered legal assistant for ESI Alger's **Citoyenneté Numérique et IA** module.

---

## ✨ Features

### ⚖️ DCIT Cyber-Law Assistant *(new)*
An AI assistant that answers questions about Algerian cyber law and digital citizenship, grounded in official legal texts via RAG (Retrieval-Augmented Generation). It never invents laws — it only answers from the documents.

- `/ask-law <question>` — Ask anything about Algerian cyber law in French
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
              ── article-aware chunking
              ── paraphrase-multilingual-MiniLM-L12-v2 (embeddings)
              ── ChromaDB (vector DB with priority metadata)
                       │
                       ▼
            rag_query.py ──── query expansion (Nmap → "accès frauduleux", etc.)
                           ── tiered retrieval: P1 → P2 → P3 fallback by distance
                           ── Groq API / llama-3.3-70b-versatile
                                      │
                                      ▼
                  bot/cogs/cyber_law_ai.py ──── /ask-law
                                             ── /law-help
```

**Priority system:** documents are tagged 1–3 at ingestion. On every query, Priority 1 sources (core cyber laws) are searched first using cosine distance scores. If similarity is too low, the search broadens to P2, then P3.

**Query expansion:** technical terms like `Nmap`, `DDoS`, `phishing`, `ransomware` are automatically mapped to their French legal equivalents (`accès frauduleux`, `atteinte au système`, etc.) before embedding, bridging the gap between student vocabulary and legal text.

---

## 📚 Knowledge Base

| Document | Description | Priority | Language |
|----------|-------------|----------|----------|
| `DZ_FR_Cybercrime Law_2009.pdf` | Loi 09-04 — cybercriminalité | 1 | FR |
| `2016_Algeria_fr_Code Penal.pdf` | Code pénal — Art. 394 bis–nonies (TIC) | 1 | FR |
| `2018_Algeria_fr_Loi n_ 18-07...pdf` | Loi 18-07 — protection des données | 1 | FR |
| `Loi n° 18-07...pdf` | Loi 18-07 — copie de cours | 1 | FR |
| `Law 20-06 Algeria.pdf` | Modifications code pénal 2020 | 2 | FR |
| `2020_Algeria_fr_Décret présidentiel n_ 20-05...pdf` | Dispositif national SSI | 2 | FR |
| `Loi n∞ 15-04...pdf` | Signature et certification électroniques | 2 | FR |
| `Penal Procedure Code 2021 Update.pdf` | Pôle pénal TIC 2021 | 2 | FR |
| `2010_en_League of Arab States Convention...pdf` | Convention arabe cybercriminalité | 2 | EN |
| `Loi organique n° 12-05...pdf` | Loi sur l'information 2012 | 3 | FR |

> **Note:** The full Penal Code is 362 pages. Only pages 108–140 (TIC articles) are ingested to avoid noise from unrelated criminal law.

---

## 🚀 Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YoucefDjenfi/DCIT-project.git
cd DCIT-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in DISCORD_TOKEN and GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Add PDFs to `knowledge_base/`

Place all legal text PDFs in `knowledge_base/`. All documents should be in French. For non-French PDFs, translate using [DeepL](https://www.deepl.com) before adding.

Update `document_priorities.py` if you add new files — filenames must match exactly.

### 4. Build the vector database

```bash
python ingest.py
```

This runs once and creates `chroma_db/`. Re-run whenever you add or update PDFs.

### 5. Run the bot

```bash
python main.py
```

Slash commands are synced automatically on startup. The first run may take ~30 seconds longer while the embedding model loads.

> **Note:** The events, quiz, and reminder features require a PostgreSQL database (`DB_URL` in `.env`). The RAG assistant works without it.

---

## 💬 Commands

### ⚖️ DCIT Legal Assistant

| Command | Description |
|---------|-------------|
| `/ask-law <question>` | Ask about Algerian cyber law (in French) |
| `/law-help` | What the bot knows and how to use it |

**Example queries:**
```
/ask-law Quelles sont les sanctions pour accès frauduleux à un système informatique ?
/ask-law Est-il légal d'effectuer un scan Nmap sur un réseau Wi-Fi public ?
/ask-law Que dit la loi 18-07 sur la collecte de données personnelles ?
/ask-law Quels sont les droits d'une personne dont les données ont été volées ?
/ask-law Qu'est-ce que la cybercriminalité selon la loi algérienne ?
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
│   │   ├── cyber_law_ai.py       ← DCIT AI assistant (new)
│   │   ├── cyberfacts_commands.py
│   │   ├── events_commands.py
│   │   ├── banned_words.py
│   │   ├── quiz_commands.py
│   │   └── ...
│   └── bot.py
├── database/
│   ├── Repositories/
│   └── connection.py
├── knowledge_base/               ← PDFs go here (gitignored)
├── chroma_db/                    ← Auto-generated vector DB (gitignored)
├── ingest.py                     ← Run once to build the DB
├── rag_query.py                  ← RAG engine with query expansion
├── document_priorities.py        ← PDF priority tier mapping
├── config.py
├── main.py
└── .env.example
```

---

## ⚠️ Limitations

- Answers are **indicative** — not a substitute for professional legal advice.
- The bot only answers from documents in its knowledge base. It will say so clearly if it cannot find an answer.
- Events, quiz, and reminder features require a PostgreSQL connection.
- Non-French PDFs (English/Arabic) produce lower-quality retrieval. Translate to French before ingesting for best results.

---

## 🙏 Acknowledgements

- Base bot: [sara-arz/shellmates-discord-bot](https://github.com/sara-arz/shellmates-discord-bot) — Shellmates Club, ESI Alger
- Legal texts: Journal Officiel de la République Algérienne, [CYRILLA Database](https://cyrilla.org), [UNIDIR Cyber Policy Portal](https://database.cyberpolicyportal.org)
- AI inference: [Groq](https://groq.com) — Meta Llama 3.3 70B Versatile
- Embeddings: [sentence-transformers](https://www.sbert.net) — `paraphrase-multilingual-MiniLM-L12-v2`
