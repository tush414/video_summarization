# 🎓 Exam Note Summarizer — YouTube / Playlist (Detailed Notes + Flashcards)

A Streamlit web app that automatically generates **detailed, exam-ready notes** and **flashcards** from YouTube videos or playlists using OpenAI models.

---

## 🚀 Features

✅ Supports single YouTube videos and entire playlists  
✅ Generates detailed notes (~6000–9000 words) per video  
✅ Creates compact summaries before expanding notes  
✅ Flashcards generation (Q&A format)  
✅ Markdown and PDF downloads  
✅ Streamlit-based user interface  
✅ Uses yt-dlp to fetch transcripts and captions  

---

## 🧩 Requirements

Install the required Python packages before running:

```bash
pip install -r requirements.txt```
---

## 🧰 How to Run

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Run the Streamlit app:

```bash
streamlit run video_summarizer_pdf.py
```

4. Enter your **OpenAI API Key** in the sidebar.
5. Paste a YouTube video or playlist URL.
6. Click **Generate Exam Notes**.

---

## 🗂️ Output Files

- **Markdown (.md)** – Contains structured notes and flashcards  
- **PDF (.pdf)** – Clean formatted version of notes (without flashcards)

---

## ⚙️ Configuration Options

- **Model name:** Choose your OpenAI model (default: `gpt-4o-mini`)  
- **Playlist mode:** Process all videos in a playlist  
- **Generate flashcards:** Toggle Q&A creation  
- **Target words:** Control detailed note length  

---
