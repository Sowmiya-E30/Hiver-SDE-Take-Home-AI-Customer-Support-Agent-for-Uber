import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# LOAD CORRECTED GOLDEN SET
# ==========================================

df = pd.read_csv("golden_evaluation_corrected.csv")

df = df[
    df["text_customer"].notna()
    & df["intent"].notna()
].copy()

df["text_customer"] = df["text_customer"].astype(str)
df["intent"] = df["intent"].astype(str)

print("Valid messages:", len(df))
print("Number of intents:", df["intent"].nunique())


X = df["text_customer"]
y = df["intent"]


# ==========================================
# 4-FOLD CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

all_actual = []
all_predictions = []

fold_accuracies = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), 1
):

    print("\nProcessing Fold", fold)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # ======================================
    # WORD TF-IDF
    # ======================================

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=10000
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)


    # ======================================
    # CHARACTER TF-IDF
    # ======================================

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=10000
    )

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)


    # ======================================
    # COMBINE WORD + CHARACTER FEATURES
    # ======================================

    X_train_combined = hstack([
        X_train_word,
        X_train_char
    ])

    X_test_combined = hstack([
        X_test_word,
        X_test_char
    ])


    # ======================================
    # TRAIN MODEL
    # ======================================

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )

    model.fit(
        X_train_combined,
        y_train
    )


    # ======================================
    # PREDICT
    # ======================================

    y_pred = model.predict(
        X_test_combined
    )


    fold_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    fold_accuracies.append(
        fold_accuracy
    )

    print(
        "Fold Accuracy:",
        round(fold_accuracy * 100, 2),
        "%"
    )


    all_actual.extend(
        y_test.tolist()
    )

    all_predictions.extend(
        y_pred.tolist()
    )


# ==========================================
# FINAL RESULTS
# ==========================================

accuracy = accuracy_score(
    all_actual,
    all_predictions
)

macro_f1 = f1_score(
    all_actual,
    all_predictions,
    average="macro"
)


print("\n======================================")
print("CHARACTER + WORD TF-IDF")
print("======================================")

print("\nFold Accuracies:")

for i, score in enumerate(
    fold_accuracies, 1
):
    print(
        "Fold", i, ":",
        round(score * 100, 2),
        "%"
    )


print(
    "\nAverage Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Average Macro F1:",
    round(macro_f1, 3)
)


# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "actual_intent": all_actual,
    "predicted_intent": all_predictions
})

results["correct"] = (
    results["actual_intent"]
    == results["predicted_intent"]
)

results.to_csv(
    "character_word_results.csv",
    index=False
)

print("\nSaved:")
print("character_word_results.csv")

print("\nDONE!")