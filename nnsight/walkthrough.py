# %%
from collections import OrderedDict
import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)
# %%
import nnsight
from nnsight import NNsight

tiny_model = NNsight(net)

# %%
print(tiny_model)

# %%
## Getting
"""
Can get model.input and model.output
"""
input = torch.rand((1, input_size))

with tiny_model.trace(input) as tracer:
    output = tiny_model.output.save()

print(output)

# %%
"""
Here we access the output of the first layer
"""
with tiny_model.trace(input) as tracer:
    l1_output = tiny_model.layer1.output.save()

print(l1_output)

# %%
"""
Get the dictionary kwargs of the inputs
"""
with tiny_model.trace(input) as tracer:
    l2_inputs = tiny_model.layer2.inputs.save()

print(l2_inputs)

# %%
"""
USING A LARGER LANGUGAGE MODEL 

"""
from nnsight import LanguageModel

llm = LanguageModel("openai-community/gpt2", device_map="auto")

print(llm)

# %%
"""
INSPECT!
"""
with llm.trace("The Eiffel Tower is in the city of"):
    # Access the last layer using h[-1] as it's a ModuleList
    # Access the first index of .output as that's where the hidden states are.
    llm.transformer.h[-1].mlp.output[0][:] = 0

    # Logits come out of model.lm_head and we apply argmax to get the predicted token ids.
    token_ids = llm.lm_head.output.argmax(dim=-1).save()

print("\nToken IDs:", token_ids)

# Apply the tokenizer to decode the ids into words after the tracing context.
print("Prediction:", llm.tokenizer.decode(token_ids[0][-1]))

# %%
"""
Run in a batch
"""
with llm.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        # Ablate the last MLP for only this batch.
        llm.transformer.h[-1].mlp.output[0][:] = 13

        # Get the output for only the intervened on batch.
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("The Eiffel Tower is in the city of"):
        # Get the output for only the original batch.
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()


print("Original token IDs:", token_ids_original)
print("Modified token IDs:", token_ids_intervention)

print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))

# %%
with llm.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = llm.transformer.wte.output

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        llm.transformer.wte.output = embeddings
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("original prediction shape", token_ids_original[0][-1].shape)
print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))

print("modified prediction shape", token_ids_intervention[0][-1].shape)
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))

# %%
from nnsight import CONFIG
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

# %%
from nnsight import LanguageModel

# We'll never actually load the parameters locally, so no need to specify a device_map.
llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:
    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])


# %%
