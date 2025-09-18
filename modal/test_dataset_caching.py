import logging
import modal
from volume import volumes
from image import image

app = modal.App("test-dataset-caching", image=image)


retries = modal.Retries(initial_delay=0.0, max_retries=10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.function(retries=retries, volumes=volumes, timeout=3600)
def load_dataset_modal(dataset_name, dataset_config=None, *args, **kwargs):
    import datasets
    import time

    logger.info("Testing dataset caching...")

    start_time = time.time()
    if dataset_config:
        datasets.load_dataset(dataset_name, dataset_config, *args, **kwargs)
    else:
        datasets.load_dataset(dataset_name, *args, **kwargs)
    load_time = time.time() - start_time

    return load_time


@app.function(retries=retries, volumes=volumes, timeout=3600)
def test_load_from_cache():
    import datasets
    import time

    logger.info("Testing dataset loading from cache...")
    logger.info("The dataset should have already been downloaded to the cache.")

    start_time = time.time()
    datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    load_time = time.time() - start_time

    logger.info(f"Dataset loaded from cache in {load_time:.2f} seconds")

    return load_time


@app.local_entrypoint()
def main():
    # Define test runs as dictionaries
    test_runs = [
        {
            "name": "wikimedia/wikipedia",
            "dataset_name": "wikimedia/wikipedia",
            "dataset_config": "20231101.en",
            "split": "train",
        },
    ]

    for run_config in test_runs:
        run_name = run_config["name"]
        dataset_name = run_config["dataset_name"]
        dataset_config = run_config.get("dataset_config")
        split = run_config.get("split", "train")

        logger.info(f"Testing {run_name} dataset...")

        # First load
        if dataset_config:
            first_load_time = load_dataset_modal.remote(
                dataset_name,
                dataset_config,
                split=split,
                download_mode="force_redownload",
            )
            second_load_time = load_dataset_modal.remote(
                dataset_name, dataset_config, split=split
            )
        else:
            first_load_time = load_dataset_modal.remote(
                dataset_name, split=split, download_mode="force_redownload"
            )
            second_load_time = load_dataset_modal.remote(dataset_name, split=split)

        speedup = (
            first_load_time / second_load_time if second_load_time > 0 else float("inf")
        )
        logger.info(f"{run_name} cache speedup: {speedup:.2f}x")

    logger.info("Dataset caching test completed successfully!")
