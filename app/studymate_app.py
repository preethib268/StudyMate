"""
studymate_app.py  —  StudyMate (Gradio, product-grade dark UI, 3 modes)
=======================================================================
Modes (lazy-loaded so memory stays manageable):
  Precise      — mpnet embeddings + trained classifier (evaluated, F1 0.644)
  Quick        — MiniLM embeddings, similarity threshold (fast, approximate)
  Transformer  — pre-trained SQuAD 2.0 model (highest accuracy, F1 ~0.80, heavier)

Run locally:  python studymate_app.py   -> http://127.0.0.1:7860

Needs in the same folder:
  studymate_full.py, transformer_mode.py, studymate_model_emb.joblib
"""

import gradio as gr
from studymate_full import StudyMate

# ---------------------------------------------------------------------------
# Modes — each entry knows how to build its analyzer, lazily
# ---------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def _build_precise():
    return StudyMate(
        model_name="all-mpnet-base-v2",
        clf_path=str(BASE_DIR / "studymate_model_emb.joblib")
    )

def _build_quick():
    return StudyMate(model_name="all-MiniLM-L6-v2", clf_path="__none__")

def _build_transformer():
    from transformer_mode import TransformerStudyMate
    return TransformerStudyMate("deepset/tinyroberta-squad2")

MODES = {
    "Precise": _build_precise,
    "Quick": _build_quick,
    "Transformer": _build_transformer,
}
_loaded = {}


def get_model(mode_label):
    if mode_label not in _loaded:
        print(f"Loading mode: {mode_label}")
        _loaded[mode_label] = MODES[mode_label]()
    return _loaded[mode_label]


DEFAULT_MODE = "Precise"
try:
    get_model(DEFAULT_MODE)   # preload the default so first check is instant
except Exception as e:
    print(f"Preload deferred (will load on first use): {e}")

EXAMPLE_MATERIAL = (
    "Photosynthesis is the process by which green plants convert sunlight into "
    "chemical energy. It takes place in the chloroplasts, which contain the pigment "
    "chlorophyll. During photosynthesis, plants absorb carbon dioxide and release "
    "oxygen as a by-product."
)
EXAMPLE_QUESTION = "Where does photosynthesis take place?"


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------
def read_file(filepath):
    if not filepath:
        return "", None
    name = filepath.lower()
    try:
        if name.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif name.endswith(".docx"):
            from docx import Document
            text = "\n".join(p.text for p in Document(filepath).paragraphs)
        elif name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        else:
            return "", "Unsupported format. Use .txt, .docx, or .pdf — or paste the text."
    except Exception as e:
        return "", f"This file could not be read ({e}). Please paste the text instead."
    clean = (text or "").strip()
    if len(clean) < 30:
        return "", ("No readable text was found — the file may be scanned or "
                    "image-based. Please paste the text instead.")
    return clean, None


def on_upload(filepath):
    text, warning = read_file(filepath)
    if warning:
        return gr.update(), gr.update(value=f"⚠️ {warning}", visible=True)
    return (gr.update(value=text), gr.update(value="✓ Loaded.", visible=True))


def load_example():
    return (gr.update(value=EXAMPLE_MATERIAL),
            gr.update(value=EXAMPLE_QUESTION),
            gr.update(value="", visible=False))


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def meter(pct, kind):
    fill = "#7c6cf0" if kind == "yes" else "#c08a4a"
    return f"""
      <div class="sm-meter"><div class="sm-meter-fill"
        style="width:{pct}%;background:{fill}"></div></div>
      <div class="sm-meter-label">Confidence · {pct}%</div>"""


def format_result(r):
    if r["answerable"]:
        conf = r["confidence"]
        also = ""
        if r.get("also_relevant"):
            also = (f'<div class="sm-label">More context</div>'
                    f'<div class="sm-quote soft">{r["also_relevant"]}</div>')
        return f"""
        <div class="sm-result yes">
          <div class="sm-verdict"><span class="sm-dot"></span>Your material answers this</div>
          {meter(conf, "yes")}
          <div class="sm-label">Where it's answered</div>
          <div class="sm-quote">{r['supporting_sentence']}</div>
          {also}
        </div>"""
    else:
        conf = 100 - r["confidence"]
        topics = "".join(f"<span class='sm-chip'>{t}</span>" for t in r["study_topics"])
        closest = ""
        if r.get("closest_sentence"):
            closest = (f'<div class="sm-label">Closest your notes get</div>'
                       f'<div class="sm-quote soft">{r["closest_sentence"]}</div>')
        return f"""
        <div class="sm-result no">
          <div class="sm-verdict"><span class="sm-dot"></span>Your material doesn't cover this</div>
          {meter(conf, "no")}
          <div class="sm-label">What's missing</div>
          <div class="sm-why">{r['why_not']}</div>
          {closest}
          <div class="sm-label">Learn these first</div>
          <div class="sm-chips">{topics}</div>
        </div>"""


def analyze(material, question, mode_label):
    material = (material or "").strip()
    question = (question or "").strip()
    if not material:
        return ("<div class='sm-empty'>Add your notes to get started — "
                "paste them or upload a document.</div>")
    if not question:
        return "<div class='sm-empty'>Ask a question to check it against your notes.</div>"
    r = get_model(mode_label).analyze(question, material)
    return format_result(r)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
:root{--bg:#0e0f14;--panel:#16181f;--field:#13151c;--line:#272a34;--txt:#eceef3;
  --muted:#8b90a0;--faint:#5d6273;--accent:#8b7bf0;--accent-soft:rgba(139,123,240,.14);
  --good:#86b89a;--warn:#c8a06a;}
.gradio-container{max-width:1000px !important;margin:auto !important;background:var(--bg) !important;
  font-family:'Inter',sans-serif !important;color:var(--txt) !important;}
.gradio-container *{font-family:'Inter',sans-serif;}
footer{display:none !important;}
#sm-nav{display:flex;align-items:center;justify-content:space-between;padding:18px 4px 0;}
#sm-brand{display:flex;align-items:center;gap:10px;}
#sm-mark{width:26px;height:26px;border-radius:7px;background:var(--accent);display:inline-flex;
  align-items:center;justify-content:center;font-family:'Fraunces',serif;font-weight:600;
  color:#0c0c12;font-size:16px;}
#sm-brandname{font-family:'Fraunces',serif;font-weight:600;font-size:1.15rem;color:var(--txt);}
#sm-navtag{color:var(--faint);font-size:.78rem;}
#sm-hero{padding:46px 0 8px;text-align:center;}
#sm-h1{font-family:'Fraunces',serif;font-weight:600;font-size:2.7rem;line-height:1.08;
  color:var(--txt);letter-spacing:-.5px;margin:0 auto;max-width:640px;}
#sm-h1 em{font-style:italic;color:var(--accent);}
#sm-sub{color:var(--muted);font-size:1.04rem;margin:16px auto 0;max-width:520px;line-height:1.6;}
.gradio-container .gr-box,.gradio-container .block{background:transparent !important;
  border:none !important;box-shadow:none !important;}
textarea,input[type=text]{background:var(--field) !important;color:var(--txt) !important;
  border:1px solid var(--line) !important;border-radius:10px !important;font-size:.97rem !important;
  line-height:1.55 !important;}
textarea::placeholder,input::placeholder{color:var(--faint) !important;}
textarea:focus,input[type=text]:focus{border-color:var(--accent) !important;
  box-shadow:0 0 0 3px var(--accent-soft) !important;}
label span{color:var(--faint) !important;font-weight:600 !important;font-size:.72rem !important;
  letter-spacing:.16em !important;text-transform:uppercase;}
button.primary,#sm-go{background:var(--accent) !important;border:none !important;color:#0c0c12 !important;
  font-weight:600 !important;border-radius:10px !important;font-size:.98rem !important;transition:filter .14s ease;}
#sm-go:hover{filter:brightness(1.07);}
#sm-example{background:transparent !important;border:1px solid var(--line) !important;color:var(--muted) !important;
  font-weight:500 !important;border-radius:9px !important;font-size:.85rem !important;}
#sm-example:hover{border-color:var(--accent) !important;color:var(--txt) !important;}
.gradio-container .gr-dropdown,.gradio-container .dropdown{background:var(--field) !important;
  border:1px solid var(--line) !important;border-radius:10px !important;color:var(--txt) !important;}
.sm-result{border-radius:16px;padding:24px 26px;border:1px solid var(--line);background:var(--panel);
  animation:rise .3s ease;}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.sm-verdict{display:flex;align-items:center;gap:11px;font-family:'Fraunces',serif;font-weight:600;
  font-size:1.2rem;color:var(--txt);}
.sm-dot{width:9px;height:9px;border-radius:50%;}
.sm-result.yes .sm-dot{background:var(--good);}
.sm-result.no .sm-dot{background:var(--warn);}
.sm-meter{height:6px;border-radius:99px;background:#0c0d13;margin:18px 0 7px;overflow:hidden;}
.sm-meter-fill{height:100%;border-radius:99px;transition:width .6s cubic-bezier(.2,.8,.2,1);}
.sm-meter-label{font-size:.74rem;color:var(--muted);letter-spacing:.03em;}
.sm-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.2em;color:var(--faint);margin:20px 0 9px;}
.sm-quote{background:var(--field);border-left:2px solid var(--accent);border-radius:0 8px 8px 0;
  padding:13px 17px;line-height:1.6;color:var(--txt);font-size:.97rem;}
.sm-result.no .sm-quote{border-left-color:var(--warn);}
.sm-quote.soft{color:var(--muted);border-left-style:dotted;}
.sm-why{line-height:1.65;color:var(--txt);font-size:.98rem;}
.sm-chips{display:flex;flex-wrap:wrap;gap:8px;}
.sm-chip{background:var(--accent-soft);border:1px solid rgba(139,123,240,.3);color:#cfc7f7;
  padding:6px 14px;border-radius:8px;font-size:.85rem;text-transform:capitalize;font-weight:500;}
.sm-empty{padding:28px;text-align:center;color:var(--muted);border:1px dashed var(--line);
  border-radius:14px;background:var(--panel);font-size:.95rem;}
#sm-foot{text-align:center;color:var(--faint);font-size:.78rem;margin:24px 0 12px;line-height:1.6;}
#sm-foot b{color:var(--muted);font-weight:600;}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="StudyMate") as demo:
    gr.HTML("""
      <div id='sm-nav'>
        <div id='sm-brand'><span id='sm-mark'>S</span><span id='sm-brandname'>StudyMate</span></div>
        <span id='sm-navtag'>Answers you can trust — from your own notes</span>
      </div>
      <div id='sm-hero'>
        <h1 id='sm-h1'>Know what your notes <em>actually</em> cover.</h1>
        <p id='sm-sub'>Ask a question, and StudyMate checks it against your material —
        pointing you to the answer when it's there, and telling you what to study
        when it isn't. No guessing, no made-up answers.</p>
      </div>
    """)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            material = gr.Textbox(label="Your notes", lines=11,
                                  placeholder="Paste your notes or a textbook passage…")
            upload = gr.File(label="Upload  ·  .txt  .docx  .pdf",
                             file_types=[".txt", ".docx", ".pdf"], type="filepath")
            status = gr.Markdown(visible=False)
        with gr.Column(scale=1):
            question = gr.Textbox(label="Your question", lines=3,
                                  placeholder="e.g. Where does photosynthesis take place?")
            mode = gr.Dropdown(choices=list(MODES.keys()), value=DEFAULT_MODE, label="Mode",
                               info="Precise = mpnet · Quick = fast MiniLM · Transformer = highest accuracy (heavier)")
            with gr.Row():
                example_btn = gr.Button("Try an example", elem_id="sm-example", scale=1)
                go = gr.Button("Check", elem_id="sm-go", scale=2)
            output = gr.HTML("<div class='sm-empty'>Your answer will appear here.</div>")
    gr.HTML("<div id='sm-foot'>StudyMate only uses the material you give it — "
            "so every answer is grounded in <b>your</b> notes.</div>")

    upload.change(on_upload, inputs=upload, outputs=[material, status])
    example_btn.click(load_example, outputs=[material, question, status])
    go.click(analyze, inputs=[material, question, mode], outputs=output)
    question.submit(analyze, inputs=[material, question, mode], outputs=output)

if __name__ == "__main__":
    demo.launch()
