import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load files
historical = pd.read_csv("uber_conversations.csv")
golden = pd.read_csv("golden_evaluation_independent.csv")

print("Historical conversations:", len(historical))
print("Golden evaluation messages:", len(golden))

# Remove Golden Set conversations from historical data
golden_ids = set(golden["tweet_id_customer"].astype(str))

historical["tweet_id_customer"] = historical["tweet_id_customer"].astype(str)

historical_clean = historical[
    ~historical["tweet_id_customer"].isin(golden_ids)
].copy()

print("Historical conversations after removing Golden Set:",
      len(historical_clean))

# Clean text
historical_clean["text_customer"] = (
    historical_clean["text_customer"]
    .fillna("")
    .astype(str)
)

golden["text_customer"] = (
    golden["text_customer"]
    .fillna("")
    .astype(str)
)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=50000
)

historical_vectors = vectorizer.fit_transform(
    historical_clean["text_customer"]
)

golden_vectors = vectorizer.transform(
    golden["text_customer"]
)

results = []

for i in range(len(golden)):

    similarities = cosine_similarity(
        golden_vectors[i],
        historical_vectors
    )[0]

    best_index = similarities.argmax()

    best_similarity = similarities[best_index]

    baseline_reply = historical_clean.iloc[
        best_index
    ]["text_uber"]

    reference_reply = golden.iloc[i][
        "reference_reply_independent"
    ]

    exact_match = (
        str(baseline_reply).strip().lower()
        ==
        str(reference_reply).strip().lower()
    )

    results.append({
        "tweet_id_customer": golden.iloc[i]["tweet_id_customer"],
        "text_customer": golden.iloc[i]["text_customer"],
        "intent": golden.iloc[i]["intent"],
        "reference_reply": reference_reply,
        "baseline_reply": baseline_reply,
        "similarity": best_similarity,
        "exact_match": exact_match
    })

    if (i + 1) % 20 == 0:
        print("Processed", i + 1)

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "baseline_results_clean.csv",
    index=False
)

# Calculate metrics
exact_match_rate = (
    results_df["exact_match"].mean() * 100
)

average_similarity = (
    results_df["similarity"].mean()
)

print()
print("======================================")
print("   LEAKAGE-FREE BASELINE RESULTS")
print("======================================")
print("Evaluation messages:", len(results_df))
print("Exact Reply Match:",
      round(exact_match_rate, 2), "%")
print("Average Reply Similarity:",
      round(average_similarity, 4))

print()
print("Saved:")
print("baseline_results_clean.csv")