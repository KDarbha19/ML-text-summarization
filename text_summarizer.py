import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import nltk
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import kagglehub
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from rouge_score import rouge_scorer

os.environ['KAGGLE_USERNAME'] = 'kausthubhdarbha'
os.environ['KAGGLE_KEY'] = 'KGAT_b194539e027e8c84efaa3a36cf906407'

# ── 1. MPS DEVICE SETUP ───────────────────────────────────────────────────────
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ── 2. LOAD & BALANCE DATA ────────────────────────────────────────────────────
print("\nLoading Amazon Fine Food Reviews...")
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
csv_path = os.path.join(path, "Reviews.csv")

df = pd.read_csv(csv_path, nrows=170000)

pos_df = df[df['Score'] >= 4].head(30000)
neg_df = df[df['Score'] <= 2].head(30000)
print(f"Pos: {len(pos_df)} | Neg: {len(neg_df)}")

df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=0).reset_index(drop=True)
df.drop_duplicates(subset=['Text'], inplace=True)
df.dropna(axis=0, inplace=True)

# ── 3. SENTIMENT-MATCHED FILTERING ───────────────────────────────────────────
positive_summary_words = {
    'great','love','best','excellent','amazing','good','perfect','delicious',
    'wonderful','fantastic','tasty','awesome','fresh','favorite','superb','outstanding'
}
negative_summary_words = {
    'terrible','horrible','awful','bad','worst','disgusting','disappointed','waste',
    'poor','gross','nasty','bland','avoid','overpriced','stale','tasteless'
}

def summary_matches_score(row):
    summary_lower = str(row['Summary']).lower()
    if row['Score'] >= 4:
        return not any(w in summary_lower for w in negative_summary_words)
    else:
        return not any(w in summary_lower for w in positive_summary_words)

df = df[df.apply(summary_matches_score, axis=1)]

pos_count = len(df[df['Score'] >= 4])
neg_count = len(df[df['Score'] <= 2])
print(f"After sentiment filter — Pos: {pos_count} | Neg: {neg_count} | Ratio: {pos_count/neg_count:.2f}")

pos_df_filtered = df[df['Score'] >= 4].sample(neg_count, random_state=0)
neg_df_filtered = df[df['Score'] <= 2]
df = pd.concat([pos_df_filtered, neg_df_filtered]).sample(frac=1, random_state=0).reset_index(drop=True)
print(f"Final balanced dataset size: {len(df)}")

# ── 4. CLEANING ───────────────────────────────────────────────────────────────
# Minimal cleaning only — T5 was pretrained on natural English so we preserve
# punctuation, case, and stopwords. Only strip HTML and expand contractions.
# Heavy cleaning like the LSTM version would hurt T5 by breaking natural language.
with open("contractions.json", "r") as f:
    contractions_dict = json.load(f)

def clean_for_t5(text):
    text = BeautifulSoup(str(text), "lxml").text
    for word, expansion in contractions_dict.items():
        text = text.replace(word, expansion)
    return text.strip()

df['clean_text']    = df['Text'].apply(clean_for_t5)
df['clean_summary'] = df['Summary'].apply(clean_for_t5)
df = df[df['clean_text'].str.len() > 0]
df = df[df['clean_summary'].str.len() > 0]

# ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
train_texts, test_texts, train_summaries, test_summaries = train_test_split(
    df['clean_text'].tolist(),
    df['clean_summary'].tolist(),
    test_size=0.2,
    random_state=0
)
print(f"\nTrain: {len(train_texts)} | Test: {len(test_texts)}")

# ── 6. LOAD T5 ────────────────────────────────────────────────────────────────
# t5-small (60M params) is the right choice for 8GB RAM with MPS.
# It runs cleanly without memory pressure and finishes in ~4-6 min per epoch.
MODEL_NAME = "t5-small"
print(f"\nLoading {MODEL_NAME} tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(device)
print(f"Model running on : {device}")
print(f"Parameters       : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── 7. TOKENIZE ───────────────────────────────────────────────────────────────
# T5 expects inputs prefixed with the task name: "summarize: <text>"
# This tells T5 which of its pretrained tasks to run.
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 20

def tokenize(input_texts, target_texts):
    prefixed = ["summarize: " + t for t in input_texts]
 
    # Tokenize inputs and targets in one call using text_target
    tokenized = tokenizer(
        prefixed,
        text_target=target_texts,
        max_length=MAX_INPUT_LEN,
        max_target_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length"
    )
 
    # Replace padding token id with -100 so the loss ignores pad tokens
    tokenized["labels"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in tokenized["labels"]
    ]
    return tokenized

print("\nTokenizing training data...")
train_tokenized = tokenize(train_texts, train_summaries)
print("Tokenizing test data...")
test_tokenized  = tokenize(test_texts, test_summaries)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset  = Dataset.from_dict(test_tokenized)
print(f"Train dataset : {len(train_dataset)} samples")
print(f"Test dataset  : {len(test_dataset)} samples")

# ── 8. TRAINING ───────────────────────────────────────────────────────────────
os.makedirs("variables_t5", exist_ok=True)

training_args = TrainingArguments(
    output_dir="variables_t5/checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=32,  # increased from 16 — MPS handles this fine
    per_device_eval_batch_size=32,
    warmup_steps=200,                # gradual LR warmup prevents early instability
    weight_decay=0.01,               # L2 regularization
    learning_rate=3e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="variables_t5/logs",
    logging_steps=50,
    fp16=False,                      # MPS does not support fp16
    use_cpu=False,                    # tell HuggingFace not to look for CUDA
    report_to="none",                # disable wandb / tensorboard
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\nStarting training...")
trainer.train()

model.save_pretrained("variables_t5/final_model")
tokenizer.save_pretrained("variables_t5/final_model")
print("\nModel and tokenizer saved to variables_t5/final_model/")

# ── 9. INFERENCE HELPER ───────────────────────────────────────────────────────
def predict_summary(review_text):
    """
    Generate a summary for a single review string.
    Uses beam search (num_beams=4) which tracks 4 candidate sequences
    simultaneously and picks the best — much more coherent than token
    by token sampling used in the LSTM version.
    """
    input_text = "summarize: " + clean_for_t5(review_text)
    input_ids = tokenizer(
        input_text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)  # send input to MPS

    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_TARGET_LEN,
        num_beams=4,            # beam search over 4 candidates
        early_stopping=True,    # stop when all beams hit end token
        no_repeat_ngram_size=2  # prevent repeating the same bigram
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ── 10. ROUGE EVALUATION ──────────────────────────────────────────────────────
print("\n--- COMPUTING ROUGE SCORES ON 200 TEST SAMPLES ---")
print("(Running inference on 200 samples, this takes a few minutes...)")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
sample_indices = np.random.choice(len(test_texts), min(200, len(test_texts)), replace=False)

for idx, i in enumerate(sample_indices):
    actual    = test_summaries[i]
    predicted = predict_summary(test_texts[i])
    if predicted and actual:
        result = scorer.score(actual, predicted)
        for k in rouge_scores:
            rouge_scores[k].append(result[k].fmeasure)
    if (idx + 1) % 50 == 0:
        print(f"  {idx + 1}/200 done...")

print(f"\nROUGE-1 : {np.mean(rouge_scores['rouge1']):.4f}  (unigram word overlap)")
print(f"ROUGE-2 : {np.mean(rouge_scores['rouge2']):.4f}  (bigram word overlap)")
print(f"ROUGE-L : {np.mean(rouge_scores['rougeL']):.4f}  (longest common subsequence)")
print(f"(ROUGE-1 above 0.3 is solid for this task)")

# ── 11. SAMPLE PREDICTIONS VS GROUND TRUTH ───────────────────────────────────
print("\n--- SAMPLE PREDICTIONS VS GROUND TRUTH ---")
for i in sample_indices[:10]:
    predicted = predict_summary(test_texts[i])
    print(f"Review    : {test_texts[i][:80]}...")
    print(f"Actual    : {test_summaries[i]}")
    print(f"Predicted : {predicted}")
    print()

# ── 12. INTERACTIVE LOOP ──────────────────────────────────────────────────────
print("\n--- INTERACTIVE MODE ---")
print("T5 works on any text, not just food reviews!")
while True:
    inp_review = input("\nEnter Review (or 'quit' to exit): ")
    if inp_review.lower() == 'quit':
        break
    summary = predict_summary(inp_review)
    print("Predicted summary:", summary)