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
