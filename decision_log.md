# Decision Log

## 1. Selected Uber as the target brand
I selected Uber because it has a large number of customer-support conversations in the TWCS dataset, providing enough historical examples for retrieval and response grounding.

## 2. Used a small custom intent taxonomy
Instead of using dozens of original dataset categories, I defined 12 practical support intents based on recurring customer problems in the Uber conversations.

## 3. Used a manually reviewed golden evaluation set
I created a 200-example evaluation set from the available Uber customer messages and manually reviewed the intent labels to make the evaluation more meaningful than relying on noisy original labels.

## 4. Used TF-IDF for historical retrieval
TF-IDF was selected as a lightweight and interpretable method for finding historically similar Uber customer-support conversations.

## 5. Retrieved multiple historical examples
The agent uses multiple similar historical conversations instead of relying on a single retrieved example. This provides the language model with more context about how Uber support previously responded.

## 6. Added rule-based signal detection
I added keyword and phrase-based signals for common support problems such as lost items, incorrect fares, cancellation fees, account compromise, driver problems, and login issues.

## 7. Used priority rules for specific issues
Specific problems are given priority over generic support or refund language. This was intended to reduce confusion between closely related intents.

## 8. Used Qwen 2.5 0.5B locally
I used the Qwen 2.5 0.5B model through Ollama so that the project could run locally without depending on a paid external LLM API.

## 9. Grounded reply generation in historical responses
The response-generation prompt includes retrieved historical Uber support examples. The model is instructed to use them as guidance and avoid inventing policies, refunds, amounts, links, or guarantees.

## 10. Added an escalation decision
The agent separately determines whether a case should be escalated. Payment, account-security, driver-safety, and refund-related issues are treated as cases that may require human investigation.

## 11. Evaluated on intent accuracy
Intent accuracy was selected as the primary automated classification metric because the golden set contains manually assigned intent labels.

## 12. Added a simple retrieval baseline
A TF-IDF historical retrieval baseline was implemented to provide a comparison against a simple non-generative approach.

## 13. Added a traditional machine-learning baseline
A character-and-word TF-IDF classifier was evaluated using cross-validation to understand how far a conventional text-classification approach could perform on the manually reviewed intent taxonomy.

## 14. Avoided using the golden set as training data
The golden evaluation examples were kept separate from the historical training/retrieval data to reduce the risk of evaluating the system on examples it had directly learned from.

## 15. Reported limitations instead of hiding weak results
The final agent achieved 37.00% intent accuracy on the 200-example golden set. I chose to report the measured result and discuss its limitations and failure modes rather than presenting an inflated or selectively chosen result.
