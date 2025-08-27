# Guideline for Probe Experiments

A guide to setting up and running experiments with trained linear probes to analyze demographic attribute representations in language models. This document provides the fundamental steps for loading probes, running inference, and designing several types of analyses.

## 1. Setup and Initialization

This section covers the initial steps of loading the language model, tokenizer, and your trained probe classifiers.

### Loading the Model and Tokenizer

First, load the pretrained model and tokenizer. The causality experiments use `meta-llama/Llama-2-13b-chat-hf`, so we'll use that as our example.

```python
# Example: Loading the model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# It's recommended to load your Hugging Face access token from a file or environment variable
# For example, from a file named 'hf_access_token.txt'
with open('hf_access_token.txt', 'r') as file:
    access_token = file.read().strip()

model_id = "meta-llama/Llama-2-13b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
model.half().to(device)
model.eval()

# Llama2's tokenizer doesn't have a default pad token. Add one to enable batching.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
```

### Loading Probe Classifiers

Your trained probes can be loaded from checkpoints using the `return_classifier_dict` utility. This function scans a directory for probe files and organizes them into a convenient dictionary.

```python
# Example: Loading probe classifiers from a checkpoint directory
from src.probes import LinearProbeClassification
from src.intervention_utils import return_classifier_dict

classifier_directory = "probe_checkpoints/controlling_probe"

# This dictionary will hold your probes, structured like: classifier_dict['age'][15] -> probe object
classifier_dict = return_classifier_dict(
    classifier_directory,
    classifier_type=LinearProbeClassification,
    logistic=True # This was set to True in the causality_test_on_age.ipynb notebook
)

print(f"Loaded probes for attributes: {list(classifier_dict.keys())}")
# Example: print(f"Layers available for 'age' probe: {list(classifier_dict['age'].keys())}")
```

## 2. Running Inference with Probes

Probes analyze the model's internal hidden states. The core workflow is to get these states for an input and then pass them to a probe.

### Getting Hidden States for Arbitrary Input

To get the hidden states for any prompt, perform a forward pass with the model and set `output_hidden_states=True`.

```python
# Example: Performing a forward pass to get hidden states
prompt = "Can you tell me about the history of the internet?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states is a tuple of tensors, one for each layer
# (plus the initial embedding layer).
hidden_states = outputs.hidden_states
print(f"Number of layers' hidden states available: {len(hidden_states)}")
```

### Applying the Probe to an Input

Once you have the hidden states, you can select a layer and apply its corresponding probe. Probes are typically trained on the representation of the _last token_, which is assumed to aggregate information from the entire sequence.

```python
# Example: Using a loaded probe to get a prediction
attribute = 'age'
target_layer = 20 # Example layer to analyze
category_labels = ['child', 'adolescent', 'adult', 'older_adult'] # For the 'age' attribute

# 1. Get the probe for the specific attribute and layer
probe = classifier_dict[attribute][target_layer]
probe.to(device)

# 2. Get hidden states for the target layer
# The shape is (batch_size, sequence_length, hidden_dim)
layer_hidden_states = hidden_states[target_layer]

# 3. Extract the representation of the last token
last_token_representation = layer_hidden_states[:, -1, :]

# 4. Apply the probe to get logits and probabilities
with torch.no_grad():
    logits = probe(last_token_representation)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_index = torch.argmax(probabilities, dim=-1).item()

print(f"Logits: {logits.squeeze().tolist()}")
print(f"Probabilities: {probabilities.squeeze().tolist()}")
print(f"Predicted class: '{category_labels[predicted_class_index]}'")
```

## 3. Example Experiments

Here are outlines for the experiments you described, building on the fundamentals above.

### Experiment 1: Probe Accuracy on a Dataset

To test probe accuracy, you need a dataset of prompts with known ground-truth labels.

**Conceptual Steps:**

1.  Define a dataset, for example, a list of dictionaries: `[{'prompt': '...', 'label': 'adult'}, ...]`.
2.  Iterate through the dataset.
3.  For each prompt, follow the steps in "Applying the Probe" to get a prediction.
4.  Compare the predicted label with the ground-truth label and aggregate the results.
5.  You can then calculate overall accuracy or use libraries like `scikit-learn` to generate a confusion matrix for more detailed analysis. This can be visualized with a bar chart showing accuracy per class.

### Experiment 2: Visualizing Token-level Effects (Logit Lens)

To see how a probe's prediction evolves across the input sequence, you can apply it to every token's representation, not just the final one. This gives insight into which tokens most influence the prediction.

**Conceptual Steps:**

1.  Get the hidden states for a specific layer: `layer_hidden_states` with shape `(1, sequence_length, hidden_dim)`.
2.  Apply the probe to all token representations at once.
    ```python
    # Squeeze the batch dimension to get shape (sequence_length, hidden_dim)
    token_representations = layer_hidden_states.squeeze(0)
    all_token_logits = probe(token_representations)
    all_token_probs = torch.softmax(all_token_logits, dim=-1)
    ```
3.  Select the probability for a target class across all tokens.
    ```python
    target_class_idx = 3 # e.g., 'older_adult'
    target_probs_over_tokens = all_token_probs[:, target_class_idx].tolist()
    ```
4.  Visualize `target_probs_over_tokens` against the input tokens to create a "logit lens" style plot.

```python
# Example: Basic visualization with Matplotlib
import matplotlib.pyplot as plt

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
plt.figure(figsize=(12, 6))
plt.bar(range(len(tokens)), target_probs_over_tokens, tick_label=tokens)
plt.xticks(rotation=90)
plt.ylabel(f"Probability of '{category_labels[target_class_idx]}'")
plt.title(f"Probe Prediction Across Input Tokens (Layer {target_layer})")
plt.show()
```

### Experiment 3: Probing Multi-turn Conversations

Probing conversations requires formatting the input so the model understands the conversational history. The `llama_v2_prompt` function handles this by structuring the turns with `[INST]` and `[/INST]` tokens.

**Conceptual Steps:**

1.  Structure your conversation as a list of message dictionaries.
2.  Use `llama_v2_prompt` to format it into a single string.
3.  Run inference as before to get hidden states for the entire conversation.
4.  You can then analyze the probe's prediction at the end of the final turn, or use the token-level approach from Experiment 2 to see how predictions change after each user or assistant message.

```python
# Example: Formatting a multi-turn conversation
from src.dataset import llama_v2_prompt

conversation = [
    {"role": "user", "content": "What's a good movie for a teenager?"},
    {"role": "assistant", "content": "I'd recommend 'Spider-Man: Into the Spider-Verse'. It's an animated film with a great story."},
    {"role": "user", "content": "Thanks! What about for someone a bit older, like a senior citizen?"}
]

# Format the conversation into a single prompt string for the model
formatted_prompt = llama_v2_prompt(conversation)

# This formatted_prompt can now be tokenized and passed to the model
# to get hidden states for analysis, as shown in previous sections.
```
