from pathlib import Path
import logging

import modal

volume = modal.Volume.from_name("cache", create_if_missing=True)

volume_path = Path("/hf")
HF_HOME_PATH = volume_path

volumes = {volume_path: volume}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "datasets",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_HOME_PATH.as_posix()})
)

app = modal.App("test-dataset-caching", image=image)


retries = modal.Retries(initial_delay=0.0, max_retries=10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.function(retries=retries, volumes=volumes)
def load_dataset_modal(*args, **kwargs):
    import datasets
    import time

    logger.info("Testing dataset caching...")

    start_time = time.time()
    datasets.load_dataset(*args, **kwargs)
    load_time = time.time() - start_time

    return load_time


@app.local_entrypoint()
def main():
    dataset_name = "wikimedia/wikipedia"
    dataset_config = "20231101.en"
    split = "train"
    first_load_time = load_dataset_modal.remote(
        dataset_name, dataset_config, split=split
    )
    second_load_time = load_dataset_modal.remote(
        dataset_name, dataset_config, split=split
    )
    speedup = (
        first_load_time / second_load_time if second_load_time > 0 else float("inf")
    )
    logger.info(f"Cache speedup: {speedup:.2f}x")

    logger.info("Dataset caching test completed successfully!")
