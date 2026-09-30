import gradio as gr
from pathlib import Path
from pypdf import PdfReader
from docx import Document

from studymate_light import StudyMateLight


analyzer = StudyMateLight()


def read_file(filepath):

    if not filepath:
        return "", None

    name = filepath.lower()

    try:

        if name.endswith(".txt"):

            with open(
                filepath,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                text = f.read()

        elif name.endswith(".docx"):

            doc = Document(filepath)

            text = "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
            )

        elif name.endswith(".pdf"):

            reader = PdfReader(filepath)

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:

            return "", (
                "Unsupported format. "
                "Use .txt, .docx, or .pdf."
            )

    except Exception as e:

        return "", f"Could not read file: {e}"

    text = (text or "").strip()

    if len(text) < 30:

        return "", (
            "No readable text was found. "
            "Please paste your notes instead."
        )

    return text, "✓ File loaded successfully."


def on_upload(filepath):

    text, message = read_file(filepath)

    if not text:

        return (
            gr.update(),
            gr.update(
                value=f"⚠️ {message}",
                visible=True
            )
        )

    return (
        gr.update(value=text),
        gr.update(
            value=f"✓ {message}",
            visible=True
        )
    )


def analyze(material, question):

    material = (material or "").strip()
    question = (question or "").strip()

    if not material:

        return """
        <div class="sm-empty">
        Add your study material first.
        </div>
        """

    if not question:

        return """
        <div class="sm-empty">
        Enter a question to check.
        </div>
        """

    result = analyzer.analyze(
        question,
        material
    )

    if "message" in result:

        return f"""
        <div class="sm-empty">
        {result["message"]}
        </div>
        """

    if result["answerable"]:

        extra = ""

        if result.get("also_relevant"):

            extra = f"""
            <div class="label">More relevant context</div>
            <div class="quote soft">
            {result["also_relevant"]}
            </div>
            """

        return f"""
        <div class="result yes">

            <h2>✓ Your material answers this</h2>

            <div class="confidence">
            Confidence: {result["confidence"]}%
            </div>

            <div class="label">
            Where it's answered
            </div>

            <div class="quote">
            {result["supporting_sentence"]}
            </div>

            {extra}

        </div>
        """

    topics = "".join(
        f"<span class='chip'>{topic}</span>"
        for topic in result.get(
            "study_topics",
            []
        )
    )

    return f"""
    <div class="result no">

        <h2>○ Your material doesn't clearly cover this</h2>

        <div class="confidence">
        Similarity: {result["confidence"]}%
        </div>

        <div class="label">
        Why
        </div>

        <div class="why">
        {result["why_not"]}
        </div>

        <div class="label">
        Closest your notes get
        </div>

        <div class="quote soft">
        {result["closest_sentence"]}
        </div>

        <div class="label">
        Topics to study
        </div>

        <div class="chips">
        {topics}
        </div>

    </div>
    """


CSS = """
body {
    background: #0e0f14;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    background: #0e0f14 !important;
    color: #eceef3 !important;
}

textarea,
input {
    background: #13151c !important;
    color: #eceef3 !important;
    border: 1px solid #272a34 !important;
    border-radius: 10px !important;
}

button {
    border-radius: 10px !important;
}

.result {
    padding: 25px;
    border-radius: 16px;
    background: #16181f;
    border: 1px solid #272a34;
}

.result h2 {
    color: #eceef3;
}

.confidence {
    color: #8b7bf0;
    margin: 15px 0;
}

.label {
    color: #8b90a0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 20px;
    margin-bottom: 8px;
}

.quote {
    background: #13151c;
    padding: 15px;
    border-left: 3px solid #8b7bf0;
    border-radius: 8px;
    line-height: 1.6;
}

.quote.soft {
    color: #8b90a0;
}

.why {
    line-height: 1.6;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.chip {
    background: #242033;
    padding: 7px 12px;
    border-radius: 8px;
    color: #cfc7f7;
}

.sm-empty {
    padding: 30px;
    text-align: center;
    color: #8b90a0;
    background: #16181f;
    border-radius: 14px;
}
"""


with gr.Blocks(
    css=CSS,
    title="StudyMate"
) as demo:

    gr.Markdown(
        """
        # 📚 StudyMate

        ### Know what your notes actually cover.

        Paste your study material, ask a question,
        and StudyMate checks whether the answer is
        present in your notes.
        """
    )

    with gr.Row():

        with gr.Column():

            material = gr.Textbox(
                label="Your notes",
                lines=12,
                placeholder=(
                    "Paste your notes or textbook passage..."
                )
            )

            upload = gr.File(
                label="Upload .txt .docx .pdf",
                file_types=[
                    ".txt",
                    ".docx",
                    ".pdf"
                ],
                type="filepath"
            )

            status = gr.Markdown(
                visible=False
            )

        with gr.Column():

            question = gr.Textbox(
                label="Your question",
                lines=4,
                placeholder=(
                    "Example: Where does photosynthesis take place?"
                )
            )

            check = gr.Button(
                "Check",
                variant="primary"
            )

    output = gr.HTML(
        """
        <div class="sm-empty">
        Your answer will appear here.
        </div>
        """
    )

    upload.change(
        on_upload,
        inputs=upload,
        outputs=[
            material,
            status
        ]
    )

    check.click(
        analyze,
        inputs=[
            material,
            question
        ],
        outputs=output
    )

    question.submit(
        analyze,
        inputs=[
            material,
            question
        ],
        outputs=output
    )
