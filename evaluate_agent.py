import pandas as pd
from agent import support_agent

# =========================
# SETTINGS
# =========================
FILE = "golden_evaluation_corrected.csv"

# Start with only 5 examples.
# After checking that it works, change 5 to None for all 200.
LIMIT = None

# =========================
# LOAD GOLDEN SET
# =========================
df = pd.read_csv(FILE)

# Keep only valid labelled examples
df = df[
    df["text_customer"].notna()
    & df["intent"].notna()
]

if LIMIT is not None:
    df = df.head(LIMIT)

print(f"Evaluating {len(df)} examples...\n")


# =========================
# RUN AGENT
# =========================
results = []

for i, row in df.iterrows():

    message = str(row["text_customer"])
    actual_intent = str(row["intent"])

    print(f"Processing {len(results) + 1}/{len(df)}")

    try:
        result = support_agent(message)

        predicted_intent = result.get("intent", "")
        reply = result.get("reply", "")
        escalate = result.get("escalate", False)
        reason = result.get("reason", "")

        correct = predicted_intent == actual_intent

        results.append({
            "message": message,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": correct,
            "reply": reply,
            "escalate": escalate,
            "reason": reason
        })

    except Exception as e:

        print("ERROR:", e)

        results.append({
            "message": message,
            "actual_intent": actual_intent,
            "predicted_intent": "ERROR",
            "intent_correct": False,
            "reply": "",
            "escalate": "",
            "reason": str(e)
        })


# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame(results)

results_df.to_csv(
    "agent_test_results.csv",
    index=False
)


# =========================
# INTENT ACCURACY
# =========================
accuracy = results_df["intent_correct"].mean() * 100

print("\n==============================")
print("EVALUATION COMPLETE")
print("==============================")

print(f"Examples tested: {len(results_df)}")
print(f"Intent Accuracy: {accuracy:.2f}%")

print("\nResults saved to:")
print("agent_test_results.csv")