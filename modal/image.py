import modal

from volume import volume_path

HF_HOME_PATH = volume_path

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "datasets",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_HOME_PATH.as_posix()})
)
