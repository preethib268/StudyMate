# StudyMate — Phase 1: Answerability Checker

An NLP tool that checks whether a student's question can be answered from their
study material (answerable vs. unanswerable detection), trained on SQuAD 2.0.

This folder contains Phase 1: the core answerability model.

------------------------------------------------------------------
## FILES IN THIS FOLDER
------------------------------------------------------------------
- train_phase1_embeddings.py  -> MAIN model (uses sentence embeddings). Run this.
- train_phase1.py             -> Simple baseline (word-matching, ~50%). For report only.
- prepare_data.py             -> Converts raw SQuAD 2.0 JSON into a clean table.
- studymate_data.csv          -> Ready-made 4000-row balanced dataset (already prepared).
- README.md                   -> This file.

NOTE: these are PROGRAM FILES (code), not documents. The steps below run them.

------------------------------------------------------------------
## STEP-BY-STEP: HOW TO RUN ON YOUR SYSTEM
------------------------------------------------------------------

### Step 1 — Install Python (skip if you already have Python 3)
- Download Python 3.10+ from https://www.python.org/downloads/
- Windows: TICK "Add Python to PATH" during install.
- Check it works:  open Terminal / Command Prompt and run:
      python --version
  (If that fails, try:  python3 --version)

### Step 2 — Put all files in one folder
- Keep this whole folder (e.g. "StudyMate") together, all files inside it.

### Step 3 — Open a terminal INSIDE this folder
- Windows: open the folder, click the address bar, type  cmd , press Enter.
- Mac: right-click the folder -> "New Terminal at Folder".

### Step 4 — Install the required libraries (one command)
      pip install scikit-learn pandas scipy joblib sentence-transformers
- If "pip" fails, try:  pip3 install ...
- Wait for it to finish (a few minutes).

### Step 5 — (Optional) Download raw SQuAD 2.0 and re-prepare data
- You ALREADY have studymate_data.csv, so you can skip this.
- If you want to regenerate it:
    1. Download dev-v2.0.json from https://rajpurkar.github.io/SQuAD-explorer/
       (the SQuAD 2.0 "dev" file). Put it in this folder.
    2. Run:
       python prepare_data.py dev-v2.0.json studymate_data.csv 4000

### Step 6 — Run the MAIN model
      python train_phase1_embeddings.py
- The FIRST run downloads a ~90 MB embedding model automatically (needs internet).
  This happens only once; after that it works offline.
- It then builds features (a few minutes — it prints progress every 500 items),
  trains the model, and prints your results.

### Step 7 — Read your results
- You'll see Accuracy, Precision, Recall, F1, and a confusion matrix.
- A trained model is saved as  studymate_model_emb.joblib  -> that's your Phase 1 output.

### Step 8 — (Optional) Run the baseline for your report
      python train_phase1.py
- Gives the ~50% word-matching result. Useful to SHOW why embeddings were needed.

------------------------------------------------------------------
## WHAT TO EXPECT
------------------------------------------------------------------
- Baseline (word-matching): ~50% accuracy. This is EXPECTED — SQuAD 2.0's
  unanswerable questions are deliberately worded to look answerable, so simple
  word-counting cannot separate them.
- Embedding model: aim for roughly high-60s to ~70s accuracy. This is a GOOD
  result on a task built to be hard. Even strong research systems only reach
  the mid-60s F1 on SQuAD 2.0.
- Your project's contribution is the HONEST, student-facing application and the
  explanation of WHY a question is unanswerable — not beating a leaderboard.

------------------------------------------------------------------
## TROUBLESHOOTING
------------------------------------------------------------------
- "python not found"  -> try  python3
- "pip not found"     -> try  pip3
- Step 6 first run needs internet (to download the model). Later runs don't.
- "Looks frozen" during embedding -> it's not; it prints progress every 500 rows. Wait.
- Re-running is fast because features are cached in features_emb.npy
  (delete that file if you change the dataset).

------------------------------------------------------------------
## NEXT (after Phase 1 works)
------------------------------------------------------------------
- Phase 2: show the supporting sentence (reuses these embeddings).
- Phase 3: explain WHY a question is unanswerable + suggest what to study.
Run Phase 1, note your accuracy, then continue to Phase 2.
