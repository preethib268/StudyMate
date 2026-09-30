import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for",
    "and", "or", "is", "are", "was", "were", "be", "been",
    "being", "what", "who", "when", "where", "which", "why",
    "how", "did", "do", "does", "this", "that", "with", "as",
    "by", "from", "into", "it", "its", "their", "his", "her",
    "they", "he", "she", "you", "i", "we", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must",
    "about", "over", "under", "between"
}


def split_sentences(text):
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", str(text))
        if s.strip()
    ]

    return sentences if sentences else [str(text)]


def content_words(text):
    return [
        w for w in re.findall(r"\b\w+\b", str(text).lower())
        if w not in STOP and len(w) > 2
    ]


def suggest_topics(question, context):
    generic = {
        "much", "many", "produced", "produce", "happen", "happens",
        "called", "used", "make", "makes", "made", "given", "give",
        "number", "amount", "kind", "type", "example", "difference",
        "between", "work", "works", "explain", "describe", "define",
        "take", "takes", "place", "places", "handle", "handles",
        "use", "uses", "get", "gets", "find", "finds", "show",
        "shows", "tell", "mean", "means", "way", "ways",
        "thing", "things"
    }

    question_words = [
        w for w in content_words(question)
        if w not in generic
    ]

    context_words = set(content_words(context))

    missing = [
        w for w in dict.fromkeys(question_words)
        if w not in context_words
    ]

    return missing[:5]


class StudyMateLight:

    def analyze(self, question, context):

        sentences = split_sentences(context)

        if not question.strip():
            return {
                "answerable": False,
                "confidence": 0,
                "message": "Please enter a question."
            }

        if not context.strip():
            return {
                "answerable": False,
                "confidence": 0,
                "message": "Please provide study material."
            }

        documents = [question] + sentences

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError:
            return {
                "answerable": False,
                "confidence": 0,
                "message": "Not enough readable text to analyze."
            }

        question_vector = matrix[0]
        sentence_vectors = matrix[1:]

        similarities = cosine_similarity(
            question_vector,
            sentence_vectors
        )[0]

        best_index = int(similarities.argmax())
        best_similarity = float(similarities[best_index])

        confidence = round(best_similarity * 100)

        # Threshold for lightweight semantic matching
        answerable = best_similarity >= 0.20

        result = {
            "answerable": answerable,
            "confidence": confidence,
            "best_similarity": round(best_similarity, 3)
        }

        if answerable:

            result["supporting_sentence"] = sentences[best_index]

            if len(similarities) > 1:
                ranked = similarities.argsort()[::-1]

                second_index = int(ranked[1])

                if similarities[second_index] >= 0.15:
                    result["also_relevant"] = sentences[second_index]

        else:

            result["closest_sentence"] = sentences[best_index]

            result["why_not"] = (
                "Your material does not clearly contain the information "
                "needed to answer this question. The closest sentence "
                "was found, but the similarity is too low to treat it "
                "as a reliable answer."
            )

            result["study_topics"] = suggest_topics(
                question,
                context
            )

        return result
