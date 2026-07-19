"""
evaluate_on_notes.py  —  Test all three methods on the student-notes dataset
============================================================================
Runs the SAME three methods from your project on YOUR real student-notes
dataset (notes_dataset.csv), producing the headline comparison for your report:

    method            accuracy   F1
    embedding (mpnet)   ...       ...
    transformer         ...       ...
    (baseline optional)

This is the evaluation of your novel contribution — how the methods perform on
real study material rather than Wikipedia.

Run:
    python evaluate_on_notes.py notes_dataset.csv

Needs in the same folder:
    studymate_full.py, transformer_mode.py, studymate_model_emb.joblib
"""

import sys
import pandas as pd


def score(preds, actuals):
    tp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(preds, actuals) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, actuals) if p == 0 and a == 1)
    n = len(actuals)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1, (tp, tn, fp, fn)


def run_method(name, analyzer, df):
    preds, actuals = [], []
    for r in df.itertuples():
        out = analyzer.analyze(r.question, r.passage)
        preds.append(1 if out["answerable"] else 0)
        actuals.append(int(r.answerable))
    acc, prec, rec, f1, cm = score(preds, actuals)
    print(f"\n--- {name} ---")
    print(f"Accuracy {acc:.3f} | Precision {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f}")
    print(f"(tp={cm[0]} tn={cm[1]} fp={cm[2]} fn={cm[3]})")
    return {"method": name, "accuracy": acc, "f1": f1}


def main(path="notes_dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["passage", "question", "answerable"])
    print(f"Loaded {len(df)} student-notes examples "
          f"({int(df['answerable'].sum())} answerable, "
          f"{int((df['answerable']==0).sum())} unanswerable)")

    results = []

    # 1) Embedding model (mpnet + trained classifier)
    from studymate_full import StudyMate
    emb = StudyMate(model_name="all-mpnet-base-v2", clf_path="studymate_model_emb.joblib")
    results.append(run_method("Embedding (mpnet)", emb, df))

    # 2) Transformer
    from transformer_mode import TransformerStudyMate
    tf = TransformerStudyMate("deepset/tinyroberta-squad2")
    results.append(run_method("Transformer (tinyroberta)", tf, df))

    # summary
    print("\n=========== SUMMARY (on real student notes) ===========")
    print(f"{'method':<28}{'accuracy':>10}{'F1':>8}")
    for r in results:
        print(f"{r['method']:<28}{r['accuracy']:>10.3f}{r['f1']:>8.3f}")
    print("\nCompare with SQuAD 2.0 (Wikipedia) results from your report to show "
          "how performance transfers to real study material.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "notes_dataset.csv"
    main(path)
