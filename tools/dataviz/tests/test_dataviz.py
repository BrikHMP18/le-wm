import json
import threading
import urllib.request

import h5py
import numpy as np
import pytest

from dataviz.reader import Recording, discover
from dataviz.server import serve


def write(path, lengths=(3, 5), chw=False, index=True):
    rows = sum(lengths)
    with h5py.File(path, "w") as file:
        shape = (rows, 3, 16, 12) if chw else (rows, 16, 12, 3)
        ramp = np.arange(rows, dtype=np.uint8)[:, None, None, None]
        file["pixels"] = ramp * np.ones(shape, np.uint8)
        file["observation"] = np.random.default_rng(0).normal(size=(rows, 4))
        action = np.ones((rows, 1), np.float32)
        action[-1] = np.nan
        file["action"] = action
        file["terminated"] = np.zeros(rows, bool)
        file["timestamp"] = np.concatenate([np.arange(n) * 0.05 for n in lengths])
        file["episode_id"] = np.repeat(np.arange(len(lengths)), lengths)
        if index:
            file["ep_len"] = np.array(lengths, np.int32)
            file["ep_offset"] = np.cumsum([0, *lengths[:-1]]).astype(np.int64)
        file.attrs["metadata_json"] = json.dumps(
            {
                "state_order": ["x", "x_dot", "theta_dot", "theta"],
                "control_dt": 0.05,
                "episodes": [{"policy": "lqr", "end_reason": "time_limit", "initial": [0]}]
                * len(lengths),
            }
        )
    return path


def test_columns_and_metadata(tmp_path):
    recording = Recording(write(tmp_path / "train.h5"))
    columns = recording.columns
    assert columns["pixels"].kind == "image"
    assert columns["observation"].names == ["x", "x_dot", "theta_dot", "theta"]
    assert {"action", "terminated", "timestamp", "episode_id"} <= set(columns)
    assert "ep_len" not in columns and recording.rows == 8
    assert recording.default_fps() == pytest.approx(20)
    assert recording.episode_attributes()[1] == {"policy": "lqr", "end_reason": "time_limit"}
    assert "episodes" not in recording.info()["metadata"]


def test_episode_slicing_and_frames(tmp_path):
    recording = Recording(write(tmp_path / "train.h5"))
    episode = recording.episode(1)
    assert episode["offset"] == 3 and episode["length"] == 5
    assert len(episode["series"]["observation"]) == 5
    assert episode["series"]["action"][-1] == [None]
    assert recording.frame("pixels", 1, 2)[0, 0, 0] == 5
    with pytest.raises(IndexError):
        recording.frame("pixels", 1, 5)


def test_chw_frames_and_fallback_index(tmp_path):
    recording = Recording(write(tmp_path / "a.h5", chw=True, index=False))
    assert recording.lengths.tolist() == [3, 5]
    assert recording.frame("pixels", 0, 1).shape == (16, 12, 3)


def test_discover_directory(tmp_path):
    write(tmp_path / "train.h5")
    (tmp_path / "sub").mkdir()
    write(tmp_path / "sub" / "test.h5")
    assert list(discover(tmp_path)) == ["sub/test.h5", "train.h5"]


def test_http_api(tmp_path):
    write(tmp_path / "train.h5")
    server = serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        get = lambda path: urllib.request.urlopen(base + path)  # noqa: E731
        assert json.load(get("/api/files")) == ["train.h5"]
        assert json.load(get("/api/info?file=train.h5"))["episodes"] == 2
        assert json.load(get("/api/episodes?file=train.h5"))["lengths"] == [3, 5]
        assert json.load(get("/api/episode?file=train.h5&ep=0"))["length"] == 3
        frame = get("/api/frame?file=train.h5&col=pixels&ep=0&t=1")
        assert frame.headers["Content-Type"] == "image/jpeg" and frame.read(2) == b"\xff\xd8"
        assert b"Dataset Viewer" in get("/").read()
        with pytest.raises(urllib.error.HTTPError) as error:
            get("/api/episode?file=train.h5&ep=9")
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
