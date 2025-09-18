from pathlib import Path
import modal

volume = modal.Volume.from_name("cache", create_if_missing=True)

volume_path = Path("/hf")

volumes = {volume_path: volume}
