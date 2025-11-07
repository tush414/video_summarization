"""
Enhanced Exam Note Summarizer with Semantic Search
- Uses OpenAI embeddings to index generated notes
- In-memory index with optional disk persistence
- Cosine similarity search & GPT-based synthesized answers
"""

import os
import re
import math
import json
import tempfile
import subprocess
from typing import List, Tuple, Dict, Any
from pathlib import Path
from io import BytesIO

import streamlit as st
import markdown2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

# PDF libs
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Numerical processing
import numpy as np

# ------------------ Helpers (timestamps, VTT parsing, captions) ------------------

def strip_timestamps(text: str) -> str:
    text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
    text = re.sub(r'<\d{1,2}:\d{2}:\d{2}\.\d{3}>', '', text)
    text = re.sub(r'<\d{1,2}:\d{2}\.\d{3}>', '', text)
    text = re.sub(r'<[^>]*\d{1,2}:\d{2}[^>]*>', '', text)
    return text

def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}:{s:02d}"

def vtt_timestamp_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(':')
        if len(parts) == 3:
            h = int(parts[0]); m = int(parts[1]); s = float(parts[2].replace(',', '.'))
        elif len(parts) == 2:
            h = 0; m = int(parts[0]); s = float(parts[1].replace(',', '.'))
        else:
            return 0.0
        return h * 3600 + m * 60 + s
    except Exception:
        return 0.0

def parse_vtt_to_transcript(vtt_path: str) -> str:
    items = []
    try:
        with open(vtt_path, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = [l.rstrip() for l in fh]
    except Exception:
        return ''
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            start_ts = line.split('-->')[0].strip()
            secs = vtt_timestamp_to_seconds(start_ts)
            i += 1
            texts = []
            while i < len(lines) and lines[i].strip() != '':
                if not lines[i].strip().isdigit():
                    texts.append(lines[i].strip())
                i += 1
            text = ' '.join(texts).replace('\n', ' ')
            items.append(f"[{format_timestamp(secs)}] {text}")
        else:
            i += 1
    return '\n'.join(items)

def fetch_captions(youtube_url_or_id: str, prefer_language: str = 'en') -> str:
    tmpd = tempfile.mkdtemp()
    url = youtube_url_or_id if youtube_url_or_id.startswith('http') else f'https://www.youtube.com/watch?v={youtube_url_or_id}'
    try:
        out_template = os.path.join(tmpd, '%(id)s.%(ext)s')
        cmd = [
            'yt-dlp','--skip-download','--write-subs','--write-auto-sub',
            '--sub-lang', prefer_language,'--convert-subs','vtt','-o',out_template,url
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        files = list(Path(tmpd).glob('*.vtt')) + list(Path(tmpd).glob('*.srt'))
        if files:
            files.sort()
            return parse_vtt_to_transcript(str(files[0]))
    except Exception as e:
        raise RuntimeError(f"Could not retrieve captions via yt-dlp: {e}")
    raise RuntimeError('Could not retrieve captions. Ensure yt-dlp is installed and captions exist.')

def build_documents_from_transcript(transcript_text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(transcript_text)
    return [Document(page_content=c) for c in chunks]

def list_videos_in_playlist(playlist_url: str) -> List[Tuple[str,str]]:
    cmd = ['yt-dlp','--flat-playlist','-J',playlist_url]
    result = subprocess.check_output(cmd, text=True)
    data = json.loads(result)
    videos = []
    for entry in data.get('entries', []):
        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
        title = entry.get('title', f"Video {entry['id']}")
        videos.append((video_url, title))
    return videos

# ------------------ OpenAI summarization & detailed notes ------------------

def summarize_documents_chunkwise(docs: List[Document], client: OpenAI, model: str="gpt-4o-mini") -> str:
    partial_summaries = []
    for d in docs:
        content = d.page_content.strip()
        if not content:
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a helpful assistant producing concise summaries of lecture transcript segments."},
                {"role":"user","content": f"Summarize this transcript segment in 2-4 paragraphs, focusing on main points and key concepts. Do NOT include timestamps:\n\n{strip_timestamps(content)}"}
            ],
            temperature=0.2
        )
        chunk_summary = resp.choices[0].message.content.strip()
        partial_summaries.append(chunk_summary)

    combined = "\n\n".join(partial_summaries)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a concise summarizer."},
            {"role":"user","content": f"Combine these segment summaries into a single coherent summary (3-6 paragraphs) of the lecture, focusing on main ideas. Do NOT include timestamps:\n\n{combined}"}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def create_outline_from_summary(summary: str, client: OpenAI, model: str="gpt-4o-mini", max_sections:int=10) -> List[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You produce an ordered outline of lecture topics."},
            {"role":"user","content": f"Create an ordered outline of {max_sections} or fewer section headings covering the lecture summary. Short headings 5-8 words. Do NOT include timestamps.\n\nSummary:\n{strip_timestamps(summary)}"}
        ],
        temperature=0.1
    )
    out = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in re.split(r'\n|\r|\d+\.|[-*]', out) if l.strip()]
    return lines[:max_sections]

def expand_section_to_length(heading: str, summary: str, client: OpenAI, target_words: int, model: str="gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You produce detailed, exam-ready lecture notes."},
            {"role":"user","content":
                f"Write a detailed section for the heading: \"{heading}\".\nUse the summary below. No timestamps. Include subheadings, lists, definitions, examples. Target ~{target_words} words.\n\nSummary:\n{strip_timestamps(summary)}"}
        ],
        temperature=0.15,
        max_tokens=14000
    )
    return strip_timestamps(resp.choices[0].message.content.strip())

def expand_into_detailed_notes(summary: str, client: OpenAI, total_target_words:int=7000, model:str="gpt-4o-mini") -> str:
    if total_target_words < 4000:
        num_sections = min(6, max(3, total_target_words // 800))
    else:
        num_sections = min(12, max(6, total_target_words // 700))
    outline = create_outline_from_summary(summary, client, max_sections=num_sections, model=model)
    if not outline:
        return expand_section_to_length("Detailed Notes", summary, client, total_target_words, model=model)

    words_remaining = total_target_words
    per_section_target = math.floor(total_target_words / max(1, len(outline)))
    detailed_parts = []

    for heading in outline:
        target = per_section_target if words_remaining - per_section_target >= 0 else max(200, words_remaining)
        section_text = expand_section_to_length(heading, summary, client, target_words=target, model=model)
        detailed_parts.append(f"## {heading}\n\n{section_text}\n")
        words_remaining -= len(section_text.split())
        if words_remaining <= 0:
            break

    combined = "\n\n".join(detailed_parts)
    combined = re.sub(r'\s+\n', '\n', combined).strip()
    return combined

# ------------------ Markdown → PDF (reportlab) ------------------

def markdown_to_pdf(md_text: str) -> BytesIO:
    md_no_flashcards = re.split(r'### Flashcards', md_text)[0]
    html_text = markdown2.markdown(md_no_flashcards, extras=["fenced-code-blocks"])

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    normal_style = styles['BodyText']
    heading1 = ParagraphStyle('Heading1', parent=styles['Heading1'], spaceAfter=12)
    heading2 = ParagraphStyle('Heading2', parent=styles['Heading2'], spaceAfter=10)
    heading3 = ParagraphStyle('Heading3', parent=styles['Heading3'], spaceAfter=8)

    Story = []

    lines = md_no_flashcards.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            Story.append(Spacer(1, 0.1*inch))
            continue
        if line.startswith("# "):
            Story.append(Paragraph(line[2:], heading1))
        elif line.startswith("## "):
            Story.append(Paragraph(line[3:], heading2))
        elif line.startswith("### "):
            Story.append(Paragraph(line[4:], heading3))
        elif line.startswith("- ") or line.startswith("* "):
            Story.append(Paragraph(line, normal_style))
        else:
            Story.append(Paragraph(line, normal_style))

    doc.build(Story)
    buffer.seek(0)
    return buffer

# ------------------ Semantic Search: Embedding + Indexing ------------------

def embed_texts(texts: List[str], client: OpenAI, model="text-embedding-3-small") -> List[List[float]]:
    """
    Embed texts one-by-one (simple loop). Returns list of embedding vectors.
    """
    embs = []
    for t in texts:
        resp = client.embeddings.create(model=model, input=t)
        embs.append(resp.data[0].embedding)
    return embs

def build_semantic_index(notes_dict: Dict[str, Any], client: OpenAI, embed_model="text-embedding-3-small",
                         chunk_size:int=800, chunk_overlap:int=100) -> List[Dict[str, Any]]:
    """
    Build an in-memory semantic index from session processed notes.
    Returns a list of dicts with keys: text, embedding (np.array), title, source_url
    """
    index = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for video_url, data in notes_dict.items():
        text = data["markdown"]
        title = data.get("title", "Untitled")
        chunks = splitter.split_text(text)
        # embed in batches (or individually)
        embeddings = embed_texts(chunks, client, model=embed_model)
        for chunk, emb in zip(chunks, embeddings):
            index.append({
                "text": chunk,
                "embedding": np.array(emb, dtype=np.float32),
                "title": title,
                "source_url": video_url
            })
    return index

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_search(query: str, index: List[Dict[str, Any]], client: OpenAI, embed_model="text-embedding-3-small", top_k:int=5):
    q_emb = np.array(client.embeddings.create(model=embed_model, input=query).data[0].embedding, dtype=np.float32)
    sims = [cosine_sim(q_emb, entry['embedding']) for entry in index]
    ranked_idx = np.argsort(sims)[-top_k:][::-1]
    results = []
    for i in ranked_idx:
        results.append({
            "text": index[i]['text'],
            "title": index[i]['title'],
            "source_url": index[i]['source_url'],
            "score": sims[i]
        })
    return results

def save_index_to_file(index: List[Dict[str, Any]], path: str):
    serial = []
    for e in index:
        serial.append({
            "text": e["text"],
            "title": e["title"],
            "source_url": e["source_url"],
            "embedding": e["embedding"].tolist()
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serial, fh)

def load_index_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for e in raw:
        out.append({
            "text": e["text"],
            "title": e["title"],
            "source_url": e["source_url"],
            "embedding": np.array(e["embedding"], dtype=np.float32)
        })
    return out

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Exam Note Summarizer", layout="wide")
st.title("📘 Youtube Exam Note Summarizer with Semantic Search")

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
model_choice = st.sidebar.text_input("Model name (chat) (e.g., gpt-4o-mini)", value="gpt-4o-mini")
embed_model_choice = st.sidebar.text_input("Embedding model (e.g., text-embedding-3-small)", value="text-embedding-3-small")
playlist_mode = st.sidebar.checkbox("Playlist Mode", value=False)
auto_detect_playlist = st.sidebar.checkbox("Auto-detect playlist URLs", value=True)
generate_flashcards = st.sidebar.checkbox("Generate flashcards", value=True)
target_words = st.sidebar.slider("Detailed notes total target words", min_value=3000, max_value=9000, value=7000, step=100)
auto_build_index = st.sidebar.checkbox("Auto-build semantic index after generation", value=True)

youtube_url = st.text_input("🎥 Enter YouTube Video or Playlist URL")
run_button = st.button("Generate Exam Notes")

# Session state
if "processed_notes" not in st.session_state:
    st.session_state.processed_notes = {}
if "semantic_index" not in st.session_state:
    st.session_state.semantic_index = None
if "index_path" not in st.session_state:
    st.session_state.index_path = None

# Main processing
if run_button:
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    try:
        if playlist_mode or (auto_detect_playlist and youtube_url and 'playlist' in youtube_url):
            st.info("Detected playlist URL — fetching all video links...")
            urls = list_videos_in_playlist(youtube_url)
        else:
            if not youtube_url:
                st.error("Please enter a YouTube URL or ID.")
                st.stop()
            urls = [(youtube_url, 'Single Video')]
    except Exception as e:
        st.error(f"Failed to list playlist videos: {e}")
        st.stop()

    progress = st.progress(0)
    total_videos = len(urls)
    client_for_index = OpenAI(api_key=api_key)  # separate reference for embedding/search

    for i, (url, title) in enumerate(urls, 1):
        if url in st.session_state.processed_notes:
            st.info(f"✅ Skipping already processed video: {title}")
            progress.progress(i / total_videos)
            continue

        st.write(f"### ▶ Processing ({i}/{total_videos}): {title}")
        try:
            transcript_ts = fetch_captions(url)
        except Exception as e:
            st.error(f"Failed to fetch captions for {title}: {e}")
            continue

        docs = build_documents_from_transcript(transcript_ts)
        compact_summary = summarize_documents_chunkwise(docs, client_for_index, model=model_choice)

        st.info(f"Generating detailed notes (~{target_words} words)...")
        detailed_notes = expand_into_detailed_notes(compact_summary, client_for_index, total_target_words=target_words, model=model_choice)

        flashcards_md = ""
        if generate_flashcards:
            try:
                fc_resp = client_for_index.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role":"system","content":"Generate concise flashcards from lecture summaries."},
                        {"role":"user","content": f"Create 12 flashcards (Q & A) in JSON from summary (no timestamps):\n\n{compact_summary}"}
                    ],
                    temperature=0.2
                )
                fc_text = fc_resp.choices[0].message.content.strip()
                try:
                    fc_list = json.loads(fc_text)
                    cards = fc_list[:12]
                    flashcards_md = "### Flashcards\n\n" + "\n\n".join([f"**Q:** {c.get('q','')}\n\n**A:** {c.get('a','')}" for c in cards])
                except Exception:
                    flashcards_md = "### Flashcards (raw)\n\n" + fc_text
            except Exception as e:
                st.warning(f"Flashcard generation failed: {e}")

        video_md_parts = [
            f"# Notes for {title}\n\n## Compact Summary\n{strip_timestamps(compact_summary)}\n",
            f"\n\n## Detailed Exam Notes (~{target_words} words)\n\n{detailed_notes}\n"
        ]
        if flashcards_md:
            video_md_parts.append(flashcards_md)

        video_markdown = "\n\n".join(video_md_parts)
        safe_title = re.sub(r'[^0-9A-Za-z _-]', '', title)[:100].replace(' ','_')

        # Store result in session state
        st.session_state.processed_notes[url] = {
            "title": title,
            "markdown": video_markdown,
            "filename": f"exam_notes_{i}_{safe_title}.md"
        }

        st.success(f"Finished processing: {title}")
        progress.progress(i / total_videos)
        st.divider()

    # Auto-build index if selected
    if auto_build_index:
        try:
            with st.spinner("Building semantic index..."):
                st.session_state.semantic_index = build_semantic_index(st.session_state.processed_notes, client_for_index, embed_model=embed_model_choice)
            st.success("Semantic index built!")
        except Exception as e:
            st.error(f"Failed to build semantic index: {e}")

# --- Render download buttons ---
if st.session_state.processed_notes:
    st.subheader("📥 Download Your Notes")
    for url, data in st.session_state.processed_notes.items():
        st.download_button(
            label=f"Download Notes for '{data['title']}'",
            data=data["markdown"],
            file_name=data["filename"],
            mime="text/markdown",
            key=f"download_{url}"
        )
        pdf_buffer = markdown_to_pdf(data["markdown"])
        st.download_button(
            label=f"Download PDF for '{data['title']}'",
            data=pdf_buffer,
            file_name=data["filename"].replace(".md",".pdf"),
            mime="application/pdf",
            key=f"pdf_{url}"
        )

# --- Semantic Search Controls ---
st.markdown("---")
st.subheader("🔍 Semantic Search")

if not api_key:
    st.info("Provide OpenAI API key in the sidebar to use semantic search.")
else:
    client = OpenAI(api_key=api_key)

    col1, col2 = st.columns([2,1])
    with col1:
        if st.session_state.semantic_index is None:
            if st.button("Build Semantic Index Now"):
                try:
                    with st.spinner("Building semantic index..."):
                        st.session_state.semantic_index = build_semantic_index(st.session_state.processed_notes, client, embed_model=embed_model_choice)
                        # default path
                        tmp_index_path = os.path.join(tempfile.gettempdir(), "semantic_index.json")
                        st.session_state.index_path = tmp_index_path
                    st.success("Semantic index built and stored in session.")
                except Exception as e:
                    st.error(f"Failed to build index: {e}")
        else:
            st.success(f"Semantic index ready — {len(st.session_state.semantic_index)} chunks indexed.")

    with col2:
        st.write("Index persistence")
        if st.session_state.semantic_index:
            if st.button("💾 Save Index to disk"):
                try:
                    path = st.text_input("Path to save index", value=os.path.join(tempfile.gettempdir(), "semantic_index.json"))
                    if path:
                        save_index_to_file(st.session_state.semantic_index, path)
                        st.session_state.index_path = path
                        st.success(f"Index saved to {path}")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        upload = st.file_uploader("📂 Load index file (JSON)", type=["json"])
        if upload is not None:
            try:
                # read uploaded file and load index
                tmp_path = os.path.join(tempfile.gettempdir(), f"uploaded_index_{os.getpid()}.json")
                with open(tmp_path, "wb") as fh:
                    fh.write(upload.getbuffer())
                st.session_state.semantic_index = load_index_from_file(tmp_path)
                st.session_state.index_path = tmp_path
                st.success("Index loaded from uploaded file.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    # Search UI
    if st.session_state.semantic_index:
        query = st.text_input("Ask a question about your notes:")
        top_k = st.slider("Top K results", min_value=1, max_value=10, value=5)
        use_gpt_answer = st.checkbox("Return synthesized answer (GPT)", value=True)
        if st.button("Search Notes") and query.strip():
            try:
                with st.spinner("Searching..."):
                    results = semantic_search(query, st.session_state.semantic_index, client, embed_model=embed_model_choice, top_k=top_k)
                st.write("### Top Results")
                for r in results:
                    st.markdown(f"**{r['title']}** — Similarity: `{r['score']:.3f}`")
                    st.markdown(r['text'][:1000] + ("..." if len(r['text'])>1000 else ""))
                    st.markdown(f"[Open source video]({r['source_url']})")
                    st.divider()

                if use_gpt_answer:
                    # Concatenate top chunks (be cautious about length)
                    top_texts = "\n\n".join([r['text'] for r in results])
                    prompt = (
                        f"Answer the user question using only the relevant extracted note chunks below. Use concise, exam-ready language. "
                        f"If the information is not present, say you couldn't find it in the notes.\n\n"
                        f"Question: {query}\n\nRelevant note chunks:\n{top_texts}\n\nAnswer:"
                    )
                    with st.spinner("Synthesizing answer via GPT..."):
                        ans_resp = client.chat.completions.create(
                            model=model_choice,
                            messages=[{"role":"user","content": prompt}],
                            temperature=0.2
                        )
                    answer = ans_resp.choices[0].message.content.strip()
                    st.write("### Synthesized Answer")
                    st.write(answer)
            except Exception as e:
                st.error(f"Search failed: {e}")
    else:
        st.info("No semantic index available yet. Build the index first (above) or generate notes first.")

st.markdown("---")
st.success("✅ App loaded. Generate notes, build the semantic index, then search your notes!")
