"""
transformer_mode.py  —  Transformer answerability for the StudyMate app
=======================================================================
Wraps a pre-trained SQuAD 2.0 transformer so it can be used as a third
'mode' in the app, exposing the SAME .analyze(question, context) interface
as the StudyMate class (so the app can call either one the same way).

Method: run the QA model, compare the best answer-span score against the
'no-answer' score (the [CLS] position). Higher span score -> answerable.
When answerable, we also return the extracted answer span as the supporting
evidence (Phase 2). When not, we reuse the same why-not + study-topic logic.

Loaded lazily by the app only when the user selects the Transformer mode.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# reuse the study-topic + sentence helpers from studymate_full so behaviour
# is consistent across modes
from studymate_full import split_sentences, content_words

STOP_GENERIC = {"much", "many", "produced", "produce", "explain", "describe",
                "define", "take", "takes", "place", "handle", "handles",
                "difference", "between", "work", "works", "use", "uses"}


class TransformerStudyMate:
    def __init__(self, model_name="deepset/tinyroberta-squad2"):
        print(f"Loading transformer '{model_name}' (first run downloads it)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Transformer ready.")

    def analyze(self, question, context):
        inputs = self.tok(question, context, return_tensors="pt",
                          truncation=True, max_length=384,
                          return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0]
        seq_ids = inputs.sequence_ids(0)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
        start = out.start_logits[0]
        end = out.end_logits[0]

        best_span = (start.max() + end.max()).item()
        no_answer = (start[0] + end[0]).item()
        answerable = best_span > no_answer
        # a soft confidence from the margin between span and no-answer scores
        margin = best_span - no_answer
        conf = 1 / (1 + pow(2.718, -margin / 2))   # squashed to 0..1

        if answerable:
            # extract the answer span text for supporting evidence
            s_idx = int(torch.argmax(start))
            e_idx = int(torch.argmax(end))
            if e_idx < s_idx:
                e_idx = s_idx
            # map token span back to the sentence that contains it
            char_s = int(offsets[s_idx][0]) if s_idx < len(offsets) else 0
            supporting = self._sentence_at(context, char_s)
            return {
                "answerable": True,
                "confidence": round(conf * 100),
                "supporting_sentence": supporting,
                "best_similarity": round(min(max(conf, 0), 1), 3),
            }
        else:
            why, closest = self._why_not(question, context)
            return {
                "answerable": False,
                "confidence": round(conf * 100),
                "why_not": why,
                "closest_sentence": closest,
                "study_topics": self._topics(question, context),
                "best_similarity": round(min(max(conf, 0), 1), 3),
            }

    def _sentence_at(self, context, char_index):
        pos = 0
        for sent in split_sentences(context):
            nxt = context.find(sent, pos)
            if nxt == -1:
                nxt = pos
            if nxt <= char_index <= nxt + len(sent):
                return sent
            pos = nxt + len(sent)
        # fallback: first sentence
        return split_sentences(context)[0]

    def _why_not(self, question, context):
        return ("The model did not find sufficient support for this question in "
                "your material — the passage does not appear to state the answer.",
                None)

    def _topics(self, question, context):
        q_seq = [w for w in content_words(question) if w not in STOP_GENERIC]
        ctx = set(content_words(context))
        raw = content_words(question)
        phrases = []
        for i in range(len(raw) - 1):
            w1, w2 = raw[i], raw[i + 1]
            if w1 in STOP_GENERIC or w2 in STOP_GENERIC:
                continue
            if w1 not in ctx or w2 not in ctx:
                p = f"{w1} {w2}"
                if p not in phrases:
                    phrases.append(p)
        used = set(" ".join(phrases).split())
        singles = [w for w in dict.fromkeys(q_seq) if w not in ctx and w not in used]
        singles.sort(key=len, reverse=True)
        topics = (phrases[:3] + singles[:2])[:5]
        return topics or ["(key terms appear, but the specific relationship isn't covered)"]
