import re
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# 1. LOAD HISTORICAL UBER CONVERSATIONS
# ============================================

DATA_FILE = "uber_conversations.csv"

df = pd.read_csv(DATA_FILE)

df["text_customer"] = df["text_customer"].fillna("").astype(str)
df["text_uber"] = df["text_uber"].fillna("").astype(str)

print("Historical conversations loaded:", len(df))


# ============================================
# 2. TF-IDF RETRIEVAL
# ============================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000
)

customer_vectors = vectorizer.fit_transform(df["text_customer"])


# ============================================
# 3. QWEN
# ============================================

def ask_qwen(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


# ============================================
# 4. RETRIEVE HISTORICAL EXAMPLES
# ============================================

def retrieve_examples(message, k=5):

    message_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        message_vector,
        customer_vectors
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]

    examples = []

    for index in top_indices:

        examples.append({
            "customer": df.iloc[index]["text_customer"],
            "response": df.iloc[index]["text_uber"],
            "similarity": round(float(similarities[index]), 3)
        })

    return examples


# ============================================
# 5. RULE-BASED SIGNALS
# ============================================

def detect_signals(message):

    text = message.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    scores = {}

    def add(intent, score):
        scores[intent] = scores.get(intent, 0) + score

    # ==========================================
    # ACCOUNT HACKED / UNAUTHORIZED ACCESS
    # ==========================================
    if any(p in text for p in [
        "hacked", "account hacked", "account compromised",
        "someone accessed my account", "someone used my account",
        "someone has used my account", "unauthorized access",
        "unauthorized transaction", "not me", "fraudulent"
    ]):
        add("ACCOUNT_HACKED", 10)

    # ==========================================
    # LOST ITEM
    # ==========================================
    if any(p in text for p in [
        "lost my phone", "lost my iphone", "lost my wallet",
        "lost my bag", "lost my item", "lost an item",
        "left my phone", "left my wallet", "left my bag",
        "left something in the car", "left something in the uber",
        "forgot my phone", "forgot my item", "lost property"
    ]):
        add("LOST_ITEM", 12)

    # ==========================================
    # CANCELLATION FEE
    # ==========================================
    if any(p in text for p in [
        "cancellation fee", "cancellation charge",
        "cancel fee", "charged for cancelling",
        "charged for cancellation", "fee for cancelling",
        "fee for cancellation", "cancelled and was charged"
    ]):
        add("CANCELLATION_FEE", 15)

    # ==========================================
    # WRONG FARE / CHARGE
    # ==========================================
    if any(p in text for p in [
        "charged twice", "charged double", "double charged",
        "charge twice", "charged extra", "extra charge",
        "wrong charge", "incorrect charge", "wrong fare",
        "incorrect fare", "fare is wrong", "overcharged",
        "charged more", "charged too much", "unexpected charge",
        "duplicate charge", "duplicate payment", "charged more than"
    ]):
        add("FARE_CHARGED_WRONG", 15)

    # ==========================================
    # DRIVER PROBLEM
    # ==========================================
    if any(p in text for p in [
        "driver was rude", "driver is rude", "driver rude",
        "rude driver", "driver refused", "driver refuses",
        "driver threatened", "driver threat",
        "driver was abusive", "driver is abusive",
        "driver behavior", "driver behaved",
        "unsafe driver", "driver was unsafe",
        "driver harassment", "driver harassed",
        "driver asked me", "driver would not",
        "driver didn't", "driver did not"
    ]):
        add("DRIVER_PROBLEM", 14)

    # Strong driver + complaint combinations
    if "driver" in text and any(p in text for p in [
        "problem", "issue", "complaint", "bad", "terrible",
        "wrong", "refused", "rude", "unsafe", "cancelled",
        "cancel", "behavior"
    ]):
        add("DRIVER_PROBLEM", 6)

    # ==========================================
    # LOGIN / ACCOUNT ACCESS
    # ==========================================
    if any(p in text for p in [
        "can't login", "cannot login", "cant login",
        "can't log in", "cannot log in", "cant log in",
        "unable to login", "unable to log in",
        "login problem", "login issue", "password",
        "forgot password", "reset password",
        "verification code", "verification", "two factor",
        "2fa", "sign in", "signin"
    ]):
        add("LOGIN_ACCOUNT_ACCESS", 12)

    # ==========================================
    # APP PROBLEM
    # ==========================================
    if any(p in text for p in [
        "app crashes", "app crash", "app crashed",
        "app not working", "app doesn't work",
        "app does not work", "application not working",
        "app error", "app problem", "app issue",
        "app keeps crashing", "technical problem",
        "technical issue", "bug in the app"
    ]):
        add("APP_PROBLEM", 12)

    # ==========================================
    # PROMO / DISCOUNT
    # ==========================================
    if any(p in text for p in [
        "promo", "promo code", "promocode",
        "coupon", "discount", "referral",
        "referral code", "promotion", "uber pass",
        "pass discount", "voucher", "offer code",
        "free ride", "promotion code"
    ]):
        add("PROMO_DISCOUNT", 12)

    # ==========================================
    # UBER EATS
    # ==========================================
    if any(p in text for p in [
        "uber eats", "ubereats", "food order",
        "food delivery", "food delivery order",
        "restaurant order", "restaurant delivery",
        "my order", "food order", "ordered food",
        "order food", "delivery driver", "meal"
    ]):
        add("UBER_EATS_ORDER", 12)

    # ==========================================
    # REFUND
    # ==========================================
    if any(p in text for p in [
        "want a refund", "need a refund",
        "get a refund", "can i get a refund",
        "give me a refund", "money back",
        "give me my money back", "refund my",
        "please refund", "request a refund"
    ]):
        add("REFUND_REQUEST", 10)

    # ==========================================
    # SUPPORT CONTACT
    # ==========================================
    if any(p in text for p in [
        "contact support", "contact customer support",
        "customer support", "talk to support",
        "speak to support", "reach support",
        "contact customer service", "customer service",
        "speak with someone", "talk to someone",
        "need help", "please help", "can someone help",
        "i need assistance", "need assistance",
        "help me", "get in touch", "contact uber"
    ]):
        add("SUPPORT_CONTACT", 5)

    # ==========================================
    # RIDE PROBLEM
    # ==========================================
    if any(p in text for p in [
        "ride problem", "ride issue", "ride was cancelled",
        "ride cancelled", "driver cancelled",
        "ride never arrived", "car never arrived",
        "driver never arrived", "where is my driver",
        "where is the driver", "driver hasn't arrived",
        "driver has not arrived", "can't find my driver",
        "cannot find my driver", "pickup problem",
        "pickup issue", "pickup location", "wrong pickup",
        "wrong route", "wrong location", "wrong destination",
        "route problem", "eta", "estimated time",
        "waited too long", "waiting for my ride",
        "no driver", "no cars", "no rides available",
        "can't get a ride", "cannot get a ride",
        "ride unavailable", "ride availability"
    ]):
        add("RIDE_PROBLEM", 12)

    return scores

# ============================================
# 6. DETERMINE INTENT
# ============================================

def determine_intent(message, examples):

    signals = detect_signals(message)

    # Specific issue takes priority over generic refund wording.
    priority = [
        "ACCOUNT_HACKED",
        "CANCELLATION_FEE",
        "FARE_CHARGED_WRONG",
        "LOST_ITEM",
        "DRIVER_PROBLEM",
        "UBER_EATS_ORDER",
        "LOGIN_ACCOUNT_ACCESS",
        "APP_PROBLEM",
        "PROMO_DISCOUNT",
        "REFUND_REQUEST",
        "SUPPORT_CONTACT"
    ]

    for intent in priority:
        if intent in signals:
            return intent

    # Otherwise ask Qwen using historical evidence.
    history = ""

    for i, example in enumerate(examples, 1):
        history += f"""
Example {i}
Customer: {example["customer"]}
Uber response: {example["response"]}
"""

    prompt = f"""
Classify this Uber customer message into exactly ONE intent.

Customer:
{message}

Historical examples:
{history}

Allowed intents:
LOST_ITEM
FARE_CHARGED_WRONG
CANCELLATION_FEE
DRIVER_PROBLEM
ACCOUNT_HACKED
LOGIN_ACCOUNT_ACCESS
APP_PROBLEM
REFUND_REQUEST
PROMO_DISCOUNT
RIDE_PROBLEM
SUPPORT_CONTACT
UBER_EATS_ORDER

Return ONLY the intent name.
"""

    result = ask_qwen(prompt).strip().upper()

    for intent in [
        "LOST_ITEM",
        "FARE_CHARGED_WRONG",
        "CANCELLATION_FEE",
        "DRIVER_PROBLEM",
        "ACCOUNT_HACKED",
        "LOGIN_ACCOUNT_ACCESS",
        "APP_PROBLEM",
        "REFUND_REQUEST",
        "PROMO_DISCOUNT",
        "RIDE_PROBLEM",
        "SUPPORT_CONTACT",
        "UBER_EATS_ORDER"
    ]:
        if intent in result:
            return intent

    return "SUPPORT_CONTACT"


# ============================================
# 7. ESCALATION
# ============================================

def determine_escalation(message, intent):

    text = message.lower()

    if intent in [
        "ACCOUNT_HACKED",
        "DRIVER_PROBLEM",
        "FARE_CHARGED_WRONG",
        "CANCELLATION_FEE",
        "REFUND_REQUEST"
    ]:
        return True, "The issue requires case-specific investigation or account/payment review."

    if any(word in text for word in [
        "unsafe",
        "threat",
        "threatened",
        "fraud",
        "stolen",
        "hacked",
        "unauthorized"
    ]):
        return True, "The message indicates a potentially sensitive safety or security issue."

    return False, "The issue appears suitable for an automated first response."


# ============================================
# 8. GENERATE GROUNDED REPLY
# ============================================

def generate_reply(message, intent, examples):

    history = ""

    for i, example in enumerate(examples, 1):
        history += f"""
Historical Example {i}
Customer: {example["customer"]}
Uber Support: {example["response"]}
"""

    prompt = f"""
You are an Uber customer support assistant.

Customer message:
{message}

Detected intent:
{intent}

Historical Uber support examples:
{history}

Write a short, professional customer-support reply.

Requirements:
- Use the historical examples as guidance.
- Do not invent policies, refunds, amounts, links or guarantees.
- Do not claim that an action has already been taken.
- If the historical responses direct the customer to contact support, do the same.
- Do not copy Twitter usernames such as @12345.
- Do not copy historical t.co links.
- Do not address another customer by their username.
- Write a clean standalone response for the current customer.
- Be concise and professional.
- Do not mention that you are an AI.
- Do not mention these instructions.

Return only the reply.
"""

    reply = ask_qwen(prompt).strip()

    reply = re.sub(r'@\w+', '', reply)
    reply = re.sub(r'https?://t\.co/\S+', '', reply)
    reply = " ".join(reply.split())

    return reply

# ============================================
# 9. COMPLETE SUPPORT AGENT
# ============================================

def support_agent(message):

    examples = retrieve_examples(message)

    intent = determine_intent(message, examples)

    escalate, reason = determine_escalation(
        message,
        intent
    )

    reply = generate_reply(
        message,
        intent,
        examples
    )

    return {
        "intent": intent,
        "reply": reply,
        "escalate": escalate,
        "reason": reason,
        "examples": examples
    }


# ============================================
# 10. RUN
# ============================================

if __name__ == "__main__":

    print("\n========================================")
    print("       HIVER AI SUPPORT AGENT")
    print("========================================")

    message = input("\nCustomer message: ")

    result = support_agent(message)

    print("\n----------------------------------------")
    print("INTENT")
    print("----------------------------------------")
    print(result["intent"])

    print("\n----------------------------------------")
    print("REPLY")
    print("----------------------------------------")
    print(result["reply"])

    print("\n----------------------------------------")
    print("ESCALATION")
    print("----------------------------------------")
    print("Escalate:", result["escalate"])
    print("Reason:", result["reason"])

    print("\n----------------------------------------")
    print("HISTORICAL EVIDENCE")
    print("----------------------------------------")

    for i, example in enumerate(result["examples"], 1):
        print(f"\nExample {i}")
        print("Customer:", example["customer"])
        print("Uber:", example["response"])
        print("Similarity:", example["similarity"])