# %%
prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"


# %%
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)
from IPython.display import clear_output
import nnsight
from nnsight import CONFIG
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected+colab+notebook"
from nnsight import LanguageModel, util

# %%
# Load gpt2
model = LanguageModel("openai-community/gpt2", device_map="auto")
clear_output()

# %%
print(model)
# %%
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
"""
Since prompts can be associated with tons of different circuits, choosing a good counterfactual prompt is important.
"""
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)

# %%
correct_index = model.tokenizer(" John")["input_ids"][0]  # includes a space
incorrect_index = model.tokenizer(" Mary")["input_ids"][0]  # includes a space

print(f"' John': {correct_index}")
print(f"' Mary': {incorrect_index}")
# %%
## FOR TESTING, what if they don't include a space
# correct_index = model.tokenizer("John")["input_ids"][0]  # includes a space
# incorrect_index = model.tokenizer("Mary")["input_ids"][0]  # includes a space

# print(f"'John': {correct_index}")
# print(f"'Mary': {incorrect_index}")

# %%
# Calculate layers
N_LAYERS = len(model.transformer.h)

# Clean run
# Start the tracing context to run the model with the clean prompt
with model.trace(clean_prompt) as tracer:
    # Get the tokens
    clean_tokens = tracer.invoker.inputs[0][0]["input_ids"][0]

    # Get hidden states of all layers in the network.
    # We index the output at 0 because it's a tuple where the first index is the hidden state.

    clean_hs = [
        model.transformer.h[layer_idx].output[0].save() for layer_idx in range(N_LAYERS)
    ]

    # Get logits from the lm_head.
    clean_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
    clean_logit_diff = (
        clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
    ).save()

# %%
# Corrupted run
with model.trace(corrupted_prompt) as tracer:
    corrupted_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
    corrupted_logit_diff = (
        corrupted_logits[0, -1, correct_index]
        - corrupted_logits[0, -1, incorrect_index]
    ).save()

# %%
# Activation Patching Intervention
ioi_patching_results = []

# Iterate through all the layers
for layer_idx in range(len(model.transformer.h)):
    _ioi_patching_results = []

    # For each token,
    for token_idx in range(len(clean_tokens)):
        # Patching corrupted run at given layer and token
        with model.trace(corrupted_prompt) as tracer:
            # Apply the patch from the clean hidden states to the corrupted hidden states.
            model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[
                layer_idx
            ][:, token_idx, :]

            patched_logits = model.lm_head.output

            patched_logit_diff = (
                patched_logits[0, -1, correct_index]
                - patched_logits[0, -1, incorrect_index]
            )

            # Calculate the improvement in the correct token after patching.
            patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                clean_logit_diff - corrupted_logit_diff
            )
            _ioi_patching_results.append(patched_result.item().save())

    ioi_patching_results.append(_ioi_patching_results)

# %%
print(ioi_patching_results)
# Plot the results
# %%

# DO IT IN ONE FORWARD PASS


N_LAYERS = len(model.transformer.h)

# Enter nnsight tracing context
with model.trace() as tracer:
    # Clean run
    with tracer.invoke(clean_prompt) as invoker:
        clean_tokens = invoker.inputs[0][0]["input_ids"][0]

        # No need to call .save() as we don't need the values after the run, just within the experiment run.
        clean_hs = [
            model.transformer.h[layer_idx].output[0].save()
            for layer_idx in range(N_LAYERS)
        ]

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()
        tracer.log(clean_logit_diff.sum())

    # Corrupted run
    with tracer.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index]
            - corrupted_logits[0, -1, incorrect_index]
        ).save()
        tracer.log(corrupted_logit_diff.sum())

    ioi_patching_results = []

    # Iterate through all the layers
    for layer_idx in range(len(model.transformer.h)):
        _ioi_patching_results = []

        # Iterate through all tokens
        for token_idx in range(len(clean_tokens)):
            # Patching corrupted run at given layer and token
            with tracer.invoke(corrupted_prompt) as invoker:
                # Set up the intervention BEFORE getting the final logits
                layer_output = model.transformer.h[layer_idx].output[0]
                layer_output[:, token_idx, :] = clean_hs[layer_idx][:, token_idx, :]

                # Now get the logits - this should reflect the patched computation
                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )
                tracer.log(patched_logit_diff.sum())
                tracer.log(corrupted_logit_diff.sum())
                tracer.log(clean_logit_diff.sum())

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )
                tracer.log(patched_result.sum())

                _ioi_patching_results.append(patched_result.item().save())

        ioi_patching_results.append(_ioi_patching_results)

# %%

from nnsight.tracing.graph import Proxy


def plot_ioi_patching_results(
    model,
    ioi_patching_results,
    x_labels,
    plot_title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
):
    ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value, Proxy)

    fig = px.imshow(
        ioi_patching_results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer", "color": "Norm. Logit Diff"},
        x=x_labels,
        title=plot_title,
    )

    return fig


# %%
print(f"Clean logit difference: {clean_logit_diff:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff:.3f}")

clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_decoded_tokens)]

print(ioi_patching_results)

fig = plot_ioi_patching_results(
    model,
    ioi_patching_results,
    token_labels,
    "Patching GPT-2-small Residual Stream on IOI task",
)
fig.show()

# %%
