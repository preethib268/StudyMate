"""
StudyMate - Lightweight Vercel Version

Uses TF-IDF + cosine similarity instead of
SentenceTransformers/PyTorch.

This keeps the StudyMate functionality while
remaining lightweight enough for Vercel.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP = set(
    """
    the a an of to in on at for and or is are was were be been being
    what who when where which why how did do does this that with as by
    from into it its their his her they he she you i we can could would
    should will shall may might must about over under between
    """.split()
)


def split_sentences(text):
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", str(text))
        if s.strip()
    ]

    return sentences if sentences else [str(text)]


def content_words(text):
    return [
        w
        for w in re.findall(r"\b\w+\b", str(text).lower())
        if w not in STOP and len(w) > 2
    ]


class StudyMate:

    def __init__(self, *args, **kwargs):
        print("Loading lightweight StudyMate model...")
        print("Using TF-IDF + cosine similarity")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )

    def analyze(self, question, context):

        question = (question or "").strip()
        context = (context or "").strip()

        if not question:
            return {
                "answerable": False,
                "confidence": 0,
                "why_not": "Please enter a question.",
                "study_topics": []
            }

        if not context:
            return {
                "answerable": False,
                "confidence": 0,
                "why_not": "Please provide study material.",
                "study_topics": []
            }

        # Split material into sentences
        sentences = split_sentences(context)

        # Combine question + sentences so they share the same vector space
        documents = [question] + sentences

        try:
            vectors = self.vectorizer.fit_transform(documents)
        except Exception:
            return {
                "answerable": False,
                "confidence": 0,
                "why_not": "The material could not be analyzed.",
                "study_topics": []
            }

        question_vector = vectors[0]
        sentence_vectors = vectors[1:]

        similarities = cosine_similarity(
            question_vector,
            sentence_vectors
        )[0]

        # Sort sentences by similarity
        order = np.argsort(similarities)[::-1]

        best_idx = int(order[0])
        best_similarity = float(similarities[best_idx])

        # Convert similarity to percentage
        confidence = round(min(best_similarity * 100, 100))

        # Threshold for answerability
        ANSWER_THRESHOLD = 0.30

        answerable = best_similarity >= ANSWER_THRESHOLD

        result = {
            "answerable": answerable,
            "confidence": confidence,
            "best_similarity": round(best_similarity, 3)
        }

        # -------------------------------------------------
        # PHASE 2 - Answer found
        # -------------------------------------------------

        if answerable:

            result["supporting_sentence"] = sentences[best_idx]

            # Find second relevant sentence
            if len(order) > 1:

                second_idx = int(order[1])
                second_similarity = float(
                    similarities[second_idx]
                )

                if second_similarity >= 0.20:

                    result["also_relevant"] = sentences[second_idx]

        # -------------------------------------------------
        # PHASE 3 - Answer not found
        # -------------------------------------------------

        else:

            closest_sentence = sentences[best_idx]

            if best_similarity < 0.10:

                reason = (
                    "This topic does not appear to be covered "
                    "in your material. None of the sentences "
                    "are closely related to your question."
                )

                closest_sentence = None

            else:

                reason = (
                    "Your material touches on this topic, "
                    "but it does not contain enough information "
                    "to answer the specific question."
                )

            result["why_not"] = reason
            result["closest_sentence"] = closest_sentence

            result["study_topics"] = self._suggest_topics(
                question,
                context
            )

        return result

    # -----------------------------------------------------
    # Suggest topics
    # -----------------------------------------------------

    def _suggest_topics(self, question, context):

        generic = {
            "much",
            "many",
            "produced",
            "produce",
            "happen",
            "happens",
            "called",
            "used",
            "make",
            "makes",
            "made",
            "given",
            "give",
            "number",
            "amount",
            "kind",
            "type",
            "example",
            "difference",
            "between",
            "work",
            "works",
            "explain",
            "describe",
            "define",
            "take",
            "takes",
            "place",
            "places",
            "handle",
            "handles",
            "use",
            "uses",
            "get",
            "gets",
            "find",
            "finds",
            "show",
            "shows",
            "tell",
            "mean",
            "means",
            "way",
            "ways",
            "thing",
            "things"
        }

        question_words = [
            w
            for w in content_words(question)
            if w not in generic
        ]

        context_words = set(content_words(context))

        missing = [
            word
            for word in dict.fromkeys(question_words)
            if word not in context_words
        ]

        # Prefer longer / more meaningful terms
        missing.sort(
            key=lambda x: len(x),
            reverse=True
        )

        topics = missing[:5]

        if topics:
            return topics

        return [
            "Review the relationship between the concepts "
            "mentioned in the question."
        ]


# ---------------------------------------------------------
# Local test
# ---------------------------------------------------------

if __name__ == "__main__":

    sm = StudyMate()

    material = (
        "Photosynthesis is the process by which green plants "
        "convert sunlight into chemical energy. "
        "It takes place in the chloroplasts, which contain "
        "the pigment chlorophyll. "
        "During photosynthesis, plants absorb carbon dioxide "
        "and release oxygen as a by-product."
    )

    questions = [
        "Where does photosynthesis take place?",
        "What gas do plants release during photosynthesis?",
        "How much ATP is produced in the Calvin cycle?"
    ]

    for question in questions:

        print("\nQuestion:", question)

        result = sm.analyze(
            question,
            material
        )

        print(result)
