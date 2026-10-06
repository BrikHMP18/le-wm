import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from teaching_cartpole.collect import Config, collect_episode, run
from teaching_cartpole.environment import CartPole, render
from teaching_cartpole.inspect import validate
from teaching_cartpole.summarize import summarize


def test_upright_equilibrium_and_linearization():
    env = CartPole([0, 0, 0, np.pi])
    np.testing.assert_allclose(env.step(0), [0, 0, 0, np.pi], atol=1e-12)
    goal, epsilon = np.array([0, 0, 0, np.pi]), 1e-6
    columns = []
    for axis in np.eye(4):
        columns.append(
            (env.dynamics(0, goal + epsilon * axis) - env.dynamics(0, goal - epsilon * axis))
            / (2 * epsilon)
        )
    a = np.column_stack(columns)
    np.testing.assert_allclose(
        a, [[0, 1, 0, 0], [0, -1.6, 0, 5.892], [0, -4.8, 0, 47.136], [0, 0, 1, 0]], atol=1e-7
    )
    env.force = 1
    np.testing.assert_allclose(env.dynamics(0, goal), [0, 1.6, 4.8, 0], atol=1e-12)


def test_reset_owns_state_and_restarts_time():
    initial = np.array([0.1, 0.2, -0.3, 1.0])
    env = CartPole(initial)
    env.step(1)
    env.reset(initial)
    np.testing.assert_array_equal(env.state, initial)
    initial[0] = 99
    assert env.state[0] == 0.1
    assert env.time == 0


def test_render_is_offscreen_and_encodes_angle_and_position():
    frames = [render(state, 64) for state in ([0, 0, 0, 0], [0, 0, 0, np.pi], [1, 0, 0, np.pi])]
    assert all(f.shape == (64, 64, 3) and f.dtype == np.uint8 for f in frames)
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


def test_collection_reproducibility_alignment_splits_and_no_overwrite(tmp_path):
    cfg = Config(episodes=6, steps=8, image_size=64, substeps=2, policy="mixed")
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = run(cfg, first)
    run(cfg, second)
    assert manifest["status"] == "complete"
    counts = validate(first, check_physics=True)
    assert sum(c["episodes"] for c in counts.values()) == 6
    for split in counts:
        with h5py.File(first / f"{split}.h5") as a, h5py.File(second / f"{split}.h5") as b:
            for key in a:
                np.testing.assert_array_equal(a[key][:], b[key][:])
    with pytest.raises(FileExistsError):
        run(cfg, first)
    assert json.loads((first / "manifest.json").read_text())["status"] == "complete"


def test_track_termination_retains_successor_and_marks_final_action():
    arrays, info = collect_episode(Config(steps=100, track_limit=0.001, image_size=64), 0)
    assert info["end_reason"] == "track_limit"
    assert len(arrays["action"]) == 2
    assert arrays["terminated"][-1]
    assert np.isnan(arrays["action"][-1, 0])


def test_inspector_rejects_wrong_actions(tmp_path):
    output = tmp_path / "collection"
    run(Config(episodes=3, steps=5, image_size=64), output)
    with h5py.File(output / "train.h5", "r+") as file:
        file["action"][0, 0] *= -1
    with pytest.raises(ValueError, match="Misaligned transition"):
        validate(output, check_physics=True)


def test_interrupted_collection_preserves_committed_episodes(tmp_path, monkeypatch):
    import teaching_cartpole.collect as collection

    def interrupt(cfg, episode_id):
        if episode_id == 1:
            raise KeyboardInterrupt
        return collect_episode(cfg, episode_id)

    monkeypatch.setattr(collection, "collect_episode", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(Config(episodes=3, steps=5, image_size=64), output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert len(manifest["episodes"]) == 1
    split = manifest["episodes"][0]["split"]
    with h5py.File(output / f"{split}.h5") as file:
        assert len(file["ep_len"]) == 1
        assert json.loads(file.attrs["metadata_json"])["status"] == "interrupted"
    with pytest.raises(ValueError, match="incomplete"):
        validate(output)


def test_parallel_collection_matches_sequential(tmp_path):
    cfg = Config(episodes=3, steps=5, image_size=64)
    run(cfg, tmp_path / "sequential")
    run(cfg, tmp_path / "parallel", workers=2)
    validate(tmp_path / "parallel", check_physics=True)
    for split in ("train", "validation", "test"):
        with h5py.File(tmp_path / "sequential" / f"{split}.h5") as a:
            with h5py.File(tmp_path / "parallel" / f"{split}.h5") as b:
                for key in a:
                    np.testing.assert_array_equal(a[key][:], b[key][:])


def test_summary_counts_frames_transitions_and_coverage(tmp_path):
    output = tmp_path / "summary"
    manifest = run(Config(episodes=3, steps=5, image_size=64, policy="mixed"), output)
    result = summarize(output)
    transitions = sum(ep["transitions"] for ep in manifest["episodes"])
    assert result["total_transitions"] == transitions
    assert result["total_frames"] == transitions + 3
    assert sum(split["episodes"] for split in result["splits"].values()) == 3
    for policy, counts in result["coverage_counts"].items():
        assert sum(counts["position"]) == result["transitions_by_policy"][policy]
        assert sum(counts["angle"]) == result["transitions_by_policy"][policy]


@pytest.mark.parametrize(
    "change",
    [
        {"physics_dt": 0},
        {"episodes": 2},
        {"seed": -1},
        {"force_limit": np.nan},
        {"track_limit": 3},
        {"val_fraction": 0.6, "test_fraction": 0.6},
    ],
)
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(), **change).validate()
