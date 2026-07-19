"""
prepare_data.py
===============
PHASE 0: Turn the raw SQuAD 2.0 JSON into a simple, clean table our model
can learn from.

Each row becomes:  question | context | answerable (1/0)

answerable = 1  -> the question CAN be answered from the context
answerable = 0  -> the question CANNOT (this is what makes the tool 'honest')

Output: studymate_data.csv
"""

import json
import csv
import sys


def convert(json_path, out_path, max_rows=None):
    data = json.load(open(json_path, encoding="utf-8"))

    rows = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            for qa in para["qas"]:
                question = qa["question"].strip()
                # is_impossible == True  means UNanswerable
                answerable = 0 if qa["is_impossible"] else 1
                rows.append({
                    "question": question,
                    "context": context,
                    "answerable": answerable,
                })

    if max_rows:
        # keep it balanced when sampling: take half from each class
        ans = [r for r in rows if r["answerable"] == 1][: max_rows // 2]
        una = [r for r in rows if r["answerable"] == 0][: max_rows // 2]
        rows = ans + una

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "context", "answerable"])
        writer.writeheader()
        writer.writerows(rows)

    ans = sum(1 for r in rows if r["answerable"] == 1)
    una = len(rows) - ans
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  answerable (1):   {ans}")
    print(f"  unanswerable (0): {una}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "dev-v2.0.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "studymate_data.csv"
    # cap the dataset so training is fast on a normal laptop; raise/remove later
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    convert(src, out, max_rows=cap)
