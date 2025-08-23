from rich.console import Console
from rich.spinner import Spinner
import time
import torch
from torch.utils.data import DataLoader, Subset, random_split
from probes import LinearProbeClassification    
import torch.nn as nn
import os
import pickle
from gender_dataset import GenderDataset, split_conversation, llama_v2_prompt

console = Console()

with console.status("[bold green]Loading packages...") as status:
    from nnsight import LanguageModel, CONFIG
    import torch
    from tqdm import tqdm
    import os
    from dotenv import load_dotenv
    import dotenv
    from transformers import BitsAndBytesConfig

with console.status("[bold green]Loading .env...") as status:
    dotenv.load_dotenv('/workspace/sandbox/.env')

with console.status("[bold green]Loading Llama model...") as status:

    # quantize model
    qcfg = BitsAndBytesConfig(load_in_8bit=True)
    llama = LanguageModel('meta-llama/Llama-2-7b-chat-hf', device_map='cuda', dispatch=True, quantization_config=qcfg, attn_implementation='eager')

    # llama = LanguageModel("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    # llama.model = llama.model.half()

os.environ["NDIF_API_KEY"] = "adbcde9f-bc87-4d14-8d47-a39a334db8c0"
os.environ["HF_TOKEN"] = "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"
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))

# Simple trace to log layer output shapes
with console.status("[bold green]Running trace to log layer shapes...") as status:
    with llama.trace("Hello world!") as tracer:
        for layer_idx in range(len(llama.model.layers)):
            layer_output = llama.model.layers[layer_idx].output
            print(f"Layer {layer_idx} output shape: {layer_output.shape}")
        
        # Also log embedding and final layer norm shapes
        embed_output = llama.model.embed_tokens.output
        
        print(f"Embedding output shape: {embed_output.shape}")


# %%
## DATASET STATS
import os

dataset_dir = "dataset"
file_count = 0
dir_count = 0

for root, dirs, files in os.walk(dataset_dir):
    print(f"Directory: {root}")
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        # print(f"  Subdirectory: {dir_path}")
        dir_count += 1
    for file in files:
        file_path = os.path.join(root, file)
        # print(f"  File: {file_path}")
        file_count += 1


# %%


class TrainerConfig:
    # optimization parameters
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # checkpoint settings

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Tokenize the inputs

dataset = GenderDataset("dataset", llama, file_limit=32)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

loss_func = nn.BCELoss()


def train(model, train_loader, optimizer, config):
    model.train()


    loss_sum = 0
    correct = 0
    tot = 0

    for batch in train_loader:
        # send the batch through the probe
        activations = batch["hidden_states"].to("cuda")
        print(activations.shape)
        label = batch["label"].to("cuda").unsqueeze(1)
        print(label)
        output = model(activations)

        loss = loss_func(output[0], label)

        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

        loss_sum += loss.item()
        correct += (output[0].argmax(dim=1) == label).sum().item()
        tot += label.size(0)

    return loss_sum / len(train_loader), correct / tot
    


def test(model, test_loader, config):
    model.eval()
    loss_sum = 0
    correct = 0
    tot = 0
    with torch.no_grad():
        for batch in test_loader:
            activations = batch["hidden_states"]
            label = batch["label"]
            output = model(activations)
            loss = loss_func(output[0], label)
            loss_sum += loss.item()
            correct += (output[0].argmax(dim=1) == label).sum().item()
            tot += label.size(0)

    return loss_sum / len(test_loader), correct / tot
    

from tqdm.auto import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
accuracy_dict = {}

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Training probes across layers...", total=40)
    
    for i in range(40):
        progress.update(task, description=f"Training probe for layer {i}")
        
        trainer_config = TrainerConfig()
        probe = LinearProbeClassification(probe_class=2, device="cuda", input_dim=4096, logistic=True)
        optimizer, scheduler = probe.configure_optimizers(trainer_config)
        best_acc = 0
        max_epoch = 50
        
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        
        epoch_pbar = tqdm(range(1, max_epoch + 1), desc=f"Layer {i} epochs", leave=False)
        
        for epoch in epoch_pbar:
            train_results = train(probe, train_loader, optimizer, trainer_config)
            test_results = test(probe, test_loader, trainer_config)
            
            train_loss, train_acc = train_results
            test_loss, test_acc = test_results
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.3f}',
                'test_loss': f'{test_loss:.3f}',
                'train_acc': f'{train_acc:.3f}',
                'test_acc': f'{test_acc:.3f}',
                'best_acc': f'{best_acc:.3f}'
            })

        accuracy_dict[f"probe_at_layer_{i}"] = best_acc
        
        table = Table(title=f"Layer {i} Training Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Best Test Accuracy", f"{best_acc:.4f}")
        table.add_row("Final Train Loss", f"{train_losses[-1]:.4f}")
        table.add_row("Final Test Loss", f"{test_losses[-1]:.4f}")
        
        console.print(table)
        progress.advance(task)

console.print(f"\n[bold green]Training completed![/bold green]")
console.print(f"Best overall accuracy: {max(accuracy_dict.values()):.4f}")


# plot




