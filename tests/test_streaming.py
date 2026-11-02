"""Streamed ray/correction math matches the dense reference on a synthetic window."""
import numpy as np
import pytest
import torch

from relift.features import interp_init
from relift.infer import VIS_GATE, observations, predict_dense, repeated_pixels
from relift.model import Corrector
from relift.streaming import MappedObservations, stream_correct


def prepared_window(root):
    T, H, W = 5, 3, 5
    yy, xx = np.mgrid[:H, :W]
    coords = np.tile(np.stack([xx, yy], -1), (T, 1, 1, 1)).astype(np.float32)
    coords[1:, ..., 0] += .5
    depth = np.arange(T * H * W, dtype=np.float32).reshape(T, H, W) / 100 + 2
    depth[0, 0, 0] = 0
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    poses[:, 0, 3] = np.arange(T) * .01
    K = np.tile(np.array([[20, 0, 2], [0, 20, 1], [0, 0, 1]], np.float32), (T, 1, 1))
    vis = np.ones((T, H, W), np.float16)
    vis[1:4, 1, 2:4] = .5
    np.savez(root / "da3_depth_data.npz", depth=depth.astype(np.float16), poses=poses, intrinsics=K)
    np.savez(root / "alltracker_tracks.npz", tracks_2d=coords.astype(np.float16), visibility=vis, confidence=vis)
    np.savez(root / "da3_frames.npz", images=np.full((T, H, W, 3), 127, np.uint8))
    return T, H, W


def test_selected_rays_keep_full_grid_rounding_and_camera_coordinates(tmp_path):
    prepared_window(tmp_path)
    dense = observations(tmp_path, 5, "cpu", depth_file="da3_depth_data.npz")
    reader = MappedObservations(tmp_path, "cpu")
    ids = np.array([14, 0, 7, 2])
    actual = reader.rays(ids)
    np.testing.assert_array_equal(actual[0], dense[0])
    for selected, full in zip(actual[1:], dense[1:]):
        expected = full.reshape((5, 15) + full.shape[3:])[:, ids]
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)


@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_streamed_corrector_matches_dense_for_ordinary_pixels_and_raw_export(tmp_path):
    T, H, W = prepared_window(tmp_path)
    torch.manual_seed(7)
    model = Corrector(dim=16, depth=1, heads=2, chunk=4).eval()
    torch.nn.init.normal_(model.head.weight, std=.01)
    torch.nn.init.normal_(model.head.bias, std=.01)
    origin, d, r, vis = observations(tmp_path, T, "cpu", depth_file="da3_depth_data.npz")
    expected = predict_dense(model, origin, d, r, vis, group=3, batch=2)
    store = stream_correct(tmp_path, tmp_path / "output", model, dict(checkpoint="synthetic"), group=3, batch=2, device="cpu")
    ids = np.arange(H * W)
    sampled = store.sample(ids)
    duplicate = repeated_pixels(r, group=3, batch=2).numpy()
    keep = np.setdiff1d(ids, duplicate)
    np.testing.assert_array_equal(store.metadata["repeated_pixel_ids"], duplicate)
    np.testing.assert_allclose(sampled["xyz"][:, keep], expected.reshape(T, -1, 3).detach().numpy()[:, keep], rtol=0, atol=1e-6)
    obs = (vis > VIS_GATE) & (r > 0)
    obs[0] = True
    Y = (origin[:, None, None, :] + d * r[..., None]) * obs[..., None]
    raw = interp_init(Y.reshape(1, T, H * W, 3), obs.reshape(1, T, H * W))[0][0]
    np.testing.assert_array_equal(sampled["lift"], raw.numpy().astype(np.float16))
    np.testing.assert_array_equal(sampled["observed"], obs.reshape(T, -1).numpy())
    resumed = stream_correct(tmp_path, tmp_path / "output", model, dict(checkpoint="synthetic"), group=3, batch=2, device="cpu")
    np.testing.assert_array_equal(resumed.sample(ids)["xyz"], sampled["xyz"])
