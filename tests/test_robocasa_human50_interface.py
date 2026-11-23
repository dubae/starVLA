"""Tests for the RoboCasa (v0.x) Human-50 eval client and task table.

Covers ``examples/simBenchmarks/Robocasa_human50``:

- the task table describes the 24-task benchmark and a 16-D state / 12-D action layout;
- ``PolicyWarper`` sends upright, resized camera views in training order, sends the
  sin/cos-encoded state only when asked to (QwenOFT changes its prompt whenever a
  ``state`` key is present), and returns ``n_action_steps`` actions in env order.

The websocket client is mocked; no policy server or robocasa simulator is required.
"""

import unittest
from unittest import mock

import numpy as np

from examples.simBenchmarks.Robocasa_human50 import human50_tasks as tasks
from examples.simBenchmarks.Robocasa_human50.eval_files import model2robocasa_human50_interface as m2r


def _fake_obs() -> dict:
    obs = {}
    for i, cam in enumerate(tasks.CAMERA_NAMES):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[0] = 255 - i  # mark the top row (as rendered, i.e. upside down)
        obs[f"{cam}_image"] = img
    for _, key, dim in tasks.STATE_LAYOUT:
        obs[key] = np.arange(dim, dtype=np.float64)
    return obs


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.requests = []

    def get_server_metadata(self) -> dict:
        return {"action_chunk_size": 16}

    def predict_action(self, query: dict) -> dict:
        self.requests.append(query)
        chunk = np.tile(np.arange(12, dtype=np.float64), (len(query["examples"]), 16, 1))
        return {"status": "ok", "data": {"actions": chunk}}

    def close(self):
        pass


class Human50TaskTableTest(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(len(tasks.HUMAN50_TASKS), 24)
        self.assertNotIn("NavigateKitchen", tasks.HUMAN50_TASKS)
        self.assertEqual(sum(dim for *_, dim in tasks.STATE_LAYOUT), 16)
        spans = [(s, e) for _, s, e in tasks.ACTION_LAYOUT]
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], 12)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))


class PolicyWarperTest(unittest.TestCase):
    def _make(self, **kwargs):
        with mock.patch.object(m2r, "WebsocketClientPolicy", _FakeClient):
            return m2r.PolicyWarper(**kwargs)

    def test_request_contents(self):
        warper = self._make(send_state=True, n_action_steps=8)
        actions = warper.step(_fake_obs(), "open the drawer")
        example = warper.client.requests[0]["examples"][0]
        self.assertEqual(example["lang"], "open the drawer")
        self.assertEqual(len(example["image"]), 3)
        self.assertEqual(example["image"][0].shape, (224, 224, 3))
        # flipped upright: the marked row is now at the bottom, views keep training order
        self.assertGreater(example["image"][0][-1].mean(), 200)
        self.assertEqual(example["image"][0][0].mean(), 0)
        self.assertEqual(example["state"].shape, (1, 32))
        # StateActionSinCosTransform is applied per state key: [sin(k1), cos(k1), sin(k2), ...]
        expected = np.concatenate([f(np.arange(d)) for *_, d in tasks.STATE_LAYOUT for f in (np.sin, np.cos)])
        np.testing.assert_allclose(example["state"][0], expected)
        self.assertEqual(actions.shape, (8, 12))

    def test_no_state(self):
        warper = self._make(send_state=False)
        warper.step(_fake_obs(), "x")
        self.assertNotIn("state", warper.client.requests[0]["examples"][0])

    def test_rejects_too_many_action_steps(self):
        with self.assertRaises(ValueError):
            self._make(n_action_steps=32)


if __name__ == "__main__":
    unittest.main()
