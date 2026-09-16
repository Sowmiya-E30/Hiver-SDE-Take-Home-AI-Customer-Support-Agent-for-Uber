# Hiver SDE Take-Home — AI Customer Support Agent for Uber

## Overview

This project implements an AI customer support agent using historical Uber customer-support conversations from the Customer Support on Twitter (TWCS) dataset.

The agent performs three main tasks:

1. Classifies an incoming customer message into a predefined support intent.
2. Retrieves historically similar Uber support conversations as evidence.
3. Generates a customer-support reply and decides whether the issue should be escalated to a human.

## Selected Brand

**Uber**

Uber was selected because the TWCS dataset contains a large historical corpus of Uber Support conversations, making it suitable for retrieval-based customer support.

## System Architecture

```text
Customer Message
       |
       v
Rule-Based Signal Detection
       |
       v
TF-IDF Historical Retrieval
       |
       v
Qwen 2.5 0.5B
       |
       +-------------> Intent Classification
       |
       +-------------> Grounded Reply
       |
       v
Escalation Rules
       |
       v
Final Support Response

## Failure Analysis

The agent achieved 37.00% intent accuracy on the 200-example golden evaluation set. The main errors came from ambiguity between closely related support intents and from the noisy nature of customer-support messages.

### Failure Mode 1: Driver problem vs. ride problem

Some messages mention a driver but are primarily about ride availability, pickup, cancellation, or arrival.

Example:
> "My driver cancelled the ride and I am still waiting."

Hypothesis:
The word "driver" strongly activates the DRIVER_PROBLEM intent, even when the actual issue is a ride operational problem. A more robust classifier should distinguish driver behaviour from general ride events.

### Failure Mode 2: Refund request vs. wrong fare

Customers may mention both an incorrect charge and a request for a refund in the same message.

Example:
> "I was charged the wrong amount and want my money back."

Hypothesis:
Both intents are semantically related. The current priority rules can select the charge-related intent even when the evaluation label focuses on the explicit refund request.

### Failure Mode 3: Generic support requests

Short messages such as "Please help" or "I need support" contain very little information about the underlying issue.

Example:
> "Please help me with this."

Hypothesis:
Without enough context, the agent has limited evidence for selecting one of the specific issue categories and may fall back to SUPPORT_CONTACT.

### Failure Mode 4: Uber Eats vs. general order/delivery language

Some messages use words such as "order", "delivery", or "driver" without explicitly mentioning Uber Eats.

Example:
> "My order hasn't arrived yet."

Hypothesis:
Generic order and delivery vocabulary can overlap with normal ride-support language. Historical Twitter messages are also noisy and may not contain enough explicit context.

### Failure Mode 5: Short and noisy Twitter messages

The dataset contains short, informal, and sometimes incomplete customer messages.

Example:
> "charged again??"

Hypothesis:
Very short messages provide insufficient semantic context. Keyword-based signals may therefore dominate, while the language model has limited historical context from which to infer the intended category.

## What Is Misleading About My Headline Number?

The headline intent accuracy of 37.00% should not be interpreted as a complete measure of the quality of the customer-support agent.

First, the evaluation contains only 200 manually reviewed examples, so it is a relatively small sample of the much larger Customer Support on Twitter dataset.

Second, the metric measures only whether the predicted intent exactly matches the assigned golden label. It does not directly measure whether the generated support reply is useful, professional, grounded in historical support behaviour, or safe.

Third, the intent labels were created using a manually defined taxonomy. Some customer messages can reasonably belong to more than one related category, particularly cases involving refunds, incorrect charges, drivers, and ride problems.

Finally, a correct intent prediction does not guarantee a correct response, while an incorrect intent label does not necessarily mean that the generated response is unusable.

Therefore, 37.00% should be treated as one diagnostic metric for intent classification rather than as an overall score for the complete support agent.
