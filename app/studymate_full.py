"""
studymate_full.py
=================
PHASE 2 + PHASE 3 for StudyMate.

Loads the model you already trained in Phase 1 (studymate_model_emb.joblib)
and turns a (study material + question) into the COMPLETE student-facing answer:

  PHASE 1  -> Is it answerable?            (uses your trained classifier)
  PHASE 2  -> WHERE is it?  (find-the-spot) show the best supporting sentence
  PHASE 3  -> If NOT answerable: WHY not + WHAT to study first

Run it two ways:
  1) Interactive demo:        python studymate_full.py
  2) Import in your app:      from studymate_full import StudyMate

Needs the same libraries as Phase 1 (sentence-transformers, scikit-learn, joblib).
The model file studymate_model_emb.joblib must be in the same folder.
IMPORTANT: set MODEL_NAME below to the SAME model you trained with
(all-mpnet-base-v2 if you kept mpnet, else all-MiniLM-L6-v2).
"""

import re
import os
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ---- MUST MATCH what you trained Phase 1 with ----
MODEL_NAME = "all-mpnet-base-v2"
CLF_PATH = "studymate_model_emb.joblib"

# thresholds for Phase 3 reasoning (tunable)
STRONG_SIM = 0.55     # a sentence this similar clearly addresses the question
WEAK_SIM = 0.35       # below this, the topic is basically absent

STOP = set("the a an of to in on at for and or is are was were be been being "
           "what who when where which why how did do does this that with as by "
           "from into it its their his her they he she you i we can could would "
           "should will shall may might must about over under between".split())


def split_sentences(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]
    return sents if sents else [str(text)]


def content_words(text):
    return [w for w in re.findall(r"\b\w+\b", str(text).lower()) if w not in STOP and len(w) > 2]


class StudyMate:
    def __init__(self, model_name=MODEL_NAME, clf_path=CLF_PATH):
        print(f"Loading embedding model '{model_name}'...")
        self.embedder = SentenceTransformer(model_name)
        self.clf = None
        if os.path.exists(clf_path):
            self.clf = joblib.load(clf_path)["model"]
            print(f"Loaded trained classifier from {clf_path}")
        else:
            print(f"WARNING: {clf_path} not found. Falling back to a similarity "
                  "threshold. Train Phase 1 first for proper accuracy.")

    # ---- the 5 features, IDENTICAL to Phase 1 training ----
    def _features(self, sims_sorted):
        max_sim = float(sims_sorted[0])
        top2_mean = float(np.mean(sims_sorted[:2]))
        mean_sim = float(np.mean(sims_sorted))
        gap = float(sims_sorted[0] - (sims_sorted[1] if len(sims_sorted) > 1 else 0))
        n_strong = float(np.sum(sims_sorted > 0.5))
        return np.array([[max_sim, top2_mean, mean_sim, gap, n_strong]])

    def analyze(self, question, context):
        sents = split_sentences(context)
        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0]
        s_emb = self.embedder.encode(sents, normalize_embeddings=True)
        sims = s_emb @ q_emb

        order = np.argsort(sims)[::-1]
        sims_sorted = sims[order]
        best_idx = int(order[0])
        best_sim = float(sims_sorted[0])

        # ----- PHASE 1: answerable? -----
        feat = self._features(sims_sorted)
        if self.clf is not None:
            answerable = bool(self.clf.predict(feat)[0])
            try:
                confidence = float(self.clf.predict_proba(feat)[0][1])
            except Exception:
                confidence = best_sim
        else:
            answerable = best_sim > 0.45
            confidence = best_sim

        result = {
            "answerable": answerable,
            "confidence": round(confidence * 100),
            "best_similarity": round(best_sim, 3),
        }

        # ----- PHASE 2: find-the-spot -----
        if answerable:
            result["supporting_sentence"] = sents[best_idx]
            # also offer the runner-up if it's close (extra context)
            if len(order) > 1 and sims_sorted[1] > STRONG_SIM:
                result["also_relevant"] = sents[int(order[1])]
        else:
            # ----- PHASE 3: why not + what to study -----
            result["why_not"], result["closest_sentence"] = self._explain_unanswerable(
                sims_sorted, sents, best_idx
            )
            result["study_topics"] = self._suggest_topics(question, context)

        return result

    def _explain_unanswerable(self, sims_sorted, sents, best_idx):
        """Classify WHY the question can't be answered from the material."""
        best_sim = sims_sorted[0]
        if best_sim < WEAK_SIM:
            reason = ("This topic does not appear to be covered in your material "
                      "at all. None of the sentences are closely related to your question.")
            closest = None
        elif best_sim < STRONG_SIM:
            reason = ("Your material touches on this topic, but it does not contain "
                      "the specific information your question asks for. The closest "
                      "sentence is related but does not fully answer it.")
            closest = sents[best_idx]
        else:
            reason = ("Your material discusses this area, but the exact answer to "
                      "your question is not clearly stated. You may be asking for a "
                      "detail that the text implies but does not state directly.")
            closest = sents[best_idx]
        return reason, closest

    def _suggest_topics(self, question, context):
        """Suggest what to study: key TOPICS (phrases where possible) from the
        question that are missing from the material.

        Strategy:
          1. Build candidate phrases (adjacent content-word pairs) from the
             question, e.g. 'calvin cycle', 'binary tree'.
          2. Keep phrases whose words don't appear in the material.
          3. Add any remaining important single missing words.
        This reads as real study topics rather than scattered words.
        """
        generic = {"much", "many", "produced", "produce", "happen", "happens",
                   "called", "used", "make", "makes", "made", "given", "give",
                   "number", "amount", "kind", "type", "example", "difference",
                   "between", "work", "works", "explain", "describe", "define",
                   "take", "takes", "place", "places", "handle", "handles",
                   "use", "uses", "get", "gets", "find", "finds", "show", "shows",
                   "tell", "mean", "means", "way", "ways", "thing", "things"}

        # ordered content words from the question (keep order, drop generics)
        q_seq = [w for w in content_words(question) if w not in generic]
        ctx_words = set(content_words(context))

        # 1) adjacent pairs from the ORIGINAL question order (phrases)
        raw_seq = content_words(question)   # includes generics for adjacency
        phrases = []
        for i in range(len(raw_seq) - 1):
            w1, w2 = raw_seq[i], raw_seq[i + 1]
            if w1 in generic or w2 in generic:
                continue
            # phrase is useful if at least one of its words is missing
            if w1 not in ctx_words or w2 not in ctx_words:
                phrase = f"{w1} {w2}"
                if phrase not in phrases:
                    phrases.append(phrase)

        # 2) single missing words not already inside a chosen phrase
        used = set(" ".join(phrases).split())
        singles = [w for w in dict.fromkeys(q_seq)
                   if w not in ctx_words and w not in used]
        singles.sort(key=len, reverse=True)

        topics = phrases[:3] + singles[:2]          # prefer phrases, allow a couple words
        topics = topics[:5]
        return topics if topics else [
            "(your question's key terms do appear, but the specific "
            "relationship isn't covered)"]


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

def _print_result(r):
    print("\n" + "-" * 55)
    if r["answerable"]:
        print(f"VERDICT: ANSWERABLE  (confidence {r['confidence']}%)")
        print(f"\n[Phase 2] Supporting sentence in your material:")
        print(f'   "{r["supporting_sentence"]}"')
        if "also_relevant" in r:
            print(f'   Also relevant: "{r["also_relevant"]}"')
    else:
        print(f"VERDICT: NOT COVERED in your material  (confidence {100 - r['confidence']}%)")
        print(f"\n[Phase 3] Why not:")
        print(f"   {r['why_not']}")
        if r.get("closest_sentence"):
            print(f'   Closest the material gets: "{r["closest_sentence"]}"')
        print(f"\n[Phase 3] Suggested topics to study first:")
        for t in r["study_topics"]:
            print(f"   - {t}")
    print("-" * 55)


if __name__ == "__main__":
    sm = StudyMate()

    # a quick built-in example so you can see it work immediately
    demo_context = (
        "Photosynthesis is the process by which green plants convert sunlight "
        "into chemical energy. It takes place in the chloroplasts, which contain "
        "the pigment chlorophyll. During photosynthesis, plants take in carbon "
        "dioxide and release oxygen as a by-product."
    )
    print("\n=== BUILT-IN DEMO ===")
    print("Study material:", demo_context)

    for q in ["Where does photosynthesis take place?",
              "What gas do plants release during photosynthesis?",
              "How much ATP is produced in the Calvin cycle?"]:
        print(f"\nQuestion: {q}")
        _print_result(sm.analyze(q, demo_context))

    # then let the user try their own
    print("\n\n=== TRY YOUR OWN (paste material once, then ask questions) ===")
    try:
        user_ctx = input("\nPaste your study material (one line), or Enter to skip:\n> ").strip()
        if user_ctx:
            while True:
                q = input("\nYour question (or 'quit'):\n> ").strip()
                if q.lower() in ("quit", "exit", ""):
                    break
                _print_result(sm.analyze(q, user_ctx))
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nDone.")
