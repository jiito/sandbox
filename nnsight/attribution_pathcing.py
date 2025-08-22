# %% 
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight

# %%
from IPython.display import clear_output
import einops
import torch
import plotly.express as px
import plotly.io as pio

pio.renderers.default = (
    "colab" if is_colab else "plotly_mimetype+notebook_connected+notebook"
)


from nnsight import LanguageModel
# %%
import nnsight
print(nnsight.__version__)

# %%
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
clear_output()
print(model)

# %%

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

# Answers are each formatted as (correct, incorrect):
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

# Tokenize clean and corrupted inputs:
clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]
# The associated corrupted input is the prompt after the current clean prompt
# for even indices, or the prompt prior to the current clean prompt for odd indices
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

# Tokenize answers for each prompt:
answer_token_indices = torch.tensor(
    [
        [model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
        for i in range(len(answers))
    ]
)
# %%

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()
# %%
clean_logits = model.trace(clean_tokens, trace=False).logits.cpu()
corrupted_logits = model.trace(corrupted_tokens, trace=False).logits.cpu()

CLEAN_BASELINE = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")

CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")
# %%
def ioi_metric(
    logits,
    answer_token_indices=answer_token_indices,
):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
# %%
