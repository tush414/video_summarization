# 🧠 Exam Note Summarizer (Detailed) — with Semantic Search

> 🎥 Turn YouTube lectures and playlists into **exam-ready detailed notes**, **flashcards**, and **semantic searchable study material** using GPT + embeddings.

---

## ✨ Features

✅ **YouTube Video / Playlist Support** — Works for single videos or full playlists.  
✅ **Automatic Caption Fetching** — Uses `yt-dlp` to retrieve subtitles.  
✅ **Timestamp-free Summaries** — Cleans transcripts and creates coherent summaries.  
✅ **Detailed Notes (~6K–9K words)** — Expanded into structured, exam-oriented notes.  
✅ **Flashcards Generation** — Optional Q&A flashcards per video.  
✅ **Markdown & PDF Download** — Notes downloadable in both formats.  
✅ **Semantic Search** — Ask natural questions across all your notes using embeddings.  
✅ **GPT-Powered Answers** — Synthesizes relevant answers from your indexed notes.  
✅ **Streamlit Frontend** — Simple, modern web interface.

---

## 🧩 Architecture Overview

```
YouTube / Playlist
     │
     ▼
[ yt-dlp ]  →  Captions (.vtt)
     │
     ▼
[ Transcript Parser + LangChain Splitter ]
     │
     ▼
[ OpenAI GPT Model ]
   ├── Compact Summary
   ├── Detailed Notes
   └── Flashcards
     │
     ▼
[ Markdown + PDF Export ]
     │
     ▼
[ Embeddings + Semantic Index ]
     │
     ▼
[ Streamlit UI: Semantic Search + GPT Answers ]
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/exam-note-summarizer.git
cd exam-note-summarizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install `yt-dlp` (if not already available globally)

```bash
pip install -U yt-dlp
```

---

## ⚙️ Configuration

Set your OpenAI API key in the Streamlit sidebar or via environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 🚀 Run the App

```bash
streamlit run exam_note_summarizer.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## 🧭 Usage Flow

1. Enter a YouTube video or playlist URL.  
2. Provide your OpenAI API key and select a model (default: `gpt-4o-mini`).  
3. Click **Generate Exam Notes**.  
4. Once done:
   - Download Markdown or PDF.
   - Build the **Semantic Index** (optional if auto-build is enabled).
5. Use **Semantic Search**:
   - Ask questions like *“Explain CNN architecture.”*
   - View top relevant chunks.
   - Optionally generate a **GPT-powered synthesized answer.**

---

## 📊 Semantic Search Details

- Uses **OpenAI embeddings** (`text-embedding-3-small`).
- Splits text into ~800-token chunks.
- Cosine similarity ranks relevant chunks.
- Optional GPT summarization of top results.

---

## 📦 File Outputs

| File Type | Description |
|------------|--------------|
| `.md` | Full markdown notes (summary, detailed notes, flashcards) |
| `.pdf` | Clean PDF version (without flashcards) |
| `.json` | Saved semantic index (optional) |

---

## 📁 Project Structure

```
exam-note-summarizer/
│
├── exam_note_summarizer.py      # Main Streamlit app
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── data/                        # (Optional) saved indexes, transcripts, PDFs
```

---

## ⚡ Requirements

See [`requirements.txt`](./requirements.txt):

```txt
streamlit>=1.38.0
openai>=1.51.0
langchain>=0.3.0
markdown2>=2.4.13
reportlab>=4.2.0
yt-dlp>=2025.1.1
numpy>=1.26.0
pandas>=2.2.2
```

---

## 🛡️ Notes & Limitations

- Captions must exist for the video.  
- Large playlists may take time.  
- API usage incurs token-based costs.  
- For larger data, use a proper vector DB.

---

