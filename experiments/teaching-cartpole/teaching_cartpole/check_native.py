"""Optional compatibility check; run in an environment with stable-worldmodel."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm


def check(output):
    results = {}
    for split in ("train", "validation", "test"):
        path = Path(output).resolve() / f"{split}.h5"
        dataset = swm.data.load_dataset(
            str(path),
            frameskip=1,
            num_steps=4,
            keys_to_load=["pixels", "action", "observation"],
            keys_to_cache=["action"],
        )
        if len(dataset) == 0:
            raise ValueError(f"{split} has no four-frame windows")
        with h5py.File(path, "r", swmr=True) as file:
            for index in {0, len(dataset) - 1}:
                episode, start = dataset.clip_indices[index]
                offset = int(file["ep_offset"][episode]) + start
                sample = dataset[index]
                expected = file["pixels"][offset : offset + 4].transpose(0, 3, 1, 2)
                np.testing.assert_array_equal(sample["pixels"].numpy(), expected)
                np.testing.assert_array_equal(
                    sample["action"].numpy(), file["action"][offset : offset + 4]
                )
                np.testing.assert_array_equal(
                    sample["observation"].numpy(), file["observation"][offset : offset + 4]
                )
                assert len(set(file["episode_id"][offset : offset + 4])) == 1
        results[split] = {
            "windows": len(dataset),
            "action_dim": dataset.get_dim("action"),
            "pixels_shape": list(sample["pixels"].shape),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(check(parser.parse_args().output), indent=2))
