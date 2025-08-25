import os
import re
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch

def split_conversation(text, user_identifier="HUMAN:", ai_identifier="ASSISTANT:"):
    user_messages = []
    assistant_messages = []

    lines = text.split("\n")

    current_user_message = ""
    current_assistant_message = ""

    for line in lines:
        line = line.lstrip(" ")
        if line.startswith(user_identifier):
            if current_assistant_message:
                assistant_messages.append(current_assistant_message.strip())
                current_assistant_message = ""
            current_user_message += line.replace(user_identifier, "").strip() + " "
        elif line.startswith(ai_identifier):
            if current_user_message:
                user_messages.append(current_user_message.strip())
                current_user_message = ""
            current_assistant_message += line.replace(ai_identifier, "").strip() + " "

    if current_user_message:
        user_messages.append(current_user_message.strip())
    if current_assistant_message:
        assistant_messages.append(current_assistant_message.strip())

    return user_messages, assistant_messages


def llama_v2_prompt(messages: list[dict], system_prompt=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    if system_prompt:
        DEFAULT_SYSTEM_PROMPT = system_prompt
    else:
        DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        messages_list.append(
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}"
        )

    return "".join(messages_list)

class GenderDataset(Dataset):
    def __init__(self, dataset_dir, llama, file_limit=None):
        self.dataset_dir = dataset_dir
        self.model = llama
        self.files = []
        self.labels = []
        self.conversations = []
        self.activations = []
        self.file_limit = file_limit
        self.load_data()

    def prompt_template(self, conversation, attribute):
        return f"{conversation} I think the {attribute} of this user is"

    def get_attribute_from_file_path(self, file_path):
        return os.path.basename(file_path).split("_")[-1].split(".")[0]

    def load_data(self):
        all_files = []
        for root, dirs, files in os.walk(self.dataset_dir):
            # regex for gender directories
            if re.match(r"^llama_gender_\d+$", os.path.basename(root)):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)

        if self.file_limit:
            all_files = all_files[:self.file_limit]

        file_p_bar = tqdm(all_files, desc="Loading conversations", position=0)
        for file_path in file_p_bar:
            # print(f"Loading {file_path}")
            self.files.append(file_path)
            conversation = self.load_conversation_from_file(file_path)
            attribute = self.get_attribute_from_file_path(file_path)

            if "HUMAN:" in conversation:
                user_msgs, ai_msgs = split_conversation(
                    conversation, "HUMAN:", "ASSISTANT:"
                )
            elif "USER:" in conversation:
                user_msgs, ai_msgs = split_conversation(
                    conversation, "USER:", "ASSISTANT:"
                )
            messages_dict = []
            for user_msg, ai_msg in zip(user_msgs, ai_msgs):
                messages_dict.append({"content": user_msg, "role": "user"})
                messages_dict.append({"content": ai_msg, "role": "assistant"})
            llama_prompt = llama_v2_prompt(messages_dict)

            llama_prompt += f" I think the {attribute} of this user is"

            activations = torch.tensor([])
            with self.model.trace(llama_prompt) as tracer:
                # save the activations from each layer
                layer_p_bar = tqdm(
                    range(len(self.model.model.layers)), 
                    desc="Saving final tokenactivations",
                    position=1,
                    leave=False
                )

                for layer_idx in layer_p_bar:
                    last_token_activations = (
                        self.model.model.layers[layer_idx].output[:,-1,:]
                        .detach()
                        .cpu()
                        .clone()
                        .to(torch.float)
                    )
                    activations = torch.cat([activations, last_token_activations], dim=0).save()
                    layer_p_bar.set_description(
                        f"Saving activations for layer {layer_idx}"
                    )
                    del last_token_activations

                layer_p_bar.close()

            self.conversations.append(llama_prompt)
            self.activations.append(activations)
            self.labels.append(attribute)

    def load_conversation_from_file(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def str_to_int(self, str):
        if str == "male":
            return 0
        elif str == "female":
            return 1
        else:
            raise ValueError(f"Invalid gender: {str}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.activations[idx],
            "label": self.str_to_int(self.labels[idx]),
        }