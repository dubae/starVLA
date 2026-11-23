"""Policy-side bridge for RoboCasa (v0.x) Human-50 evaluation (PandaOmron, single arm).

Converts a raw robosuite observation dict into a StarVLA example, queries the
websocket policy server and returns an (n_action_steps, 12) action chunk that can be
passed to ``env.step()`` as-is (the training action layout is the native env order,
see ``human50_tasks.ACTION_LAYOUT``).

Train/test consistency (mirrors ``train_files/data_registry/data_config.py``):
  - images : the three cameras, flipped to the upright (HDF5) convention, resized
             with PIL bicubic exactly like ``LeRobotSingleDataset._pack_sample``;
  - state  : 16-D in ``STATE_LAYOUT`` order, then ``[sin(s), cos(s)]`` (32-D), sent
             only when the checkpoint was trained with ``include_state: true``;
  - action : un-normalized by the server with the training statistics.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.Robocasa_human50.human50_tasks import CAMERA_NAMES, STATE_LAYOUT

DEFAULT_IMAGE_KEYS = tuple(f"video.{cam}" for cam in CAMERA_NAMES)


class PolicyWarper:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5678,
        unnorm_key: Optional[str] = None,
        image_size: Sequence[int] = (224, 224),
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        n_action_steps: int = 8,
        send_state: bool = True,
        flip_images: bool = True,
        num_steps: Optional[int] = None,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.server_metadata = self.client.get_server_metadata()
        self.unnorm_key = unnorm_key
        self.image_size = tuple(int(x) for x in image_size)
        self.image_keys = tuple(image_keys)
        self.n_action_steps = int(n_action_steps)
        # QwenOFT appends discretized state tokens to the prompt whenever `state` is
        # present, so a checkpoint trained with include_state=false must not get it.
        self.send_state = send_state
        # robosuite renders with the OpenGL convention (upside down); the HDF5 training
        # images were stored upright.
        self.flip_images = flip_images
        self.num_steps = num_steps

        chunk = int(self.server_metadata.get("action_chunk_size", self.n_action_steps))
        if self.n_action_steps > chunk:
            raise ValueError(f"n_action_steps={self.n_action_steps} exceeds the model action chunk ({chunk})")

    def _image(self, obs: Dict, key: str) -> Image.Image:
        cam = key.replace("video.", "")
        img = obs[f"{cam}_image"]
        if self.flip_images:
            img = img[::-1]
        return Image.fromarray(np.ascontiguousarray(img)).resize(self.image_size)

    @staticmethod
    def build_state(obs: Dict) -> np.ndarray:
        # StateActionSinCosTransform runs per state key and the keys are then concatenated:
        # [sin(k1), cos(k1), sin(k2), cos(k2), ...] -- NOT [sin(all), cos(all)].
        parts = []
        for _, obs_key, _ in STATE_LAYOUT:
            s = np.asarray(obs[obs_key], dtype=np.float64).reshape(-1)
            parts += [np.sin(s), np.cos(s)]
        return np.concatenate(parts)[None]  # (1, 32): one observation step, as in training

    def upright_images(self, obs: Dict) -> list[np.ndarray]:
        """Camera frames in the training orientation (used for video logging)."""
        return [np.ascontiguousarray(obs[f"{k.replace('video.', '')}_image"][::-1] if self.flip_images
                                     else obs[f"{k.replace('video.', '')}_image"]) for k in self.image_keys]

    def step(self, obs: Dict, instruction: str) -> np.ndarray:
        example = {
            "image": [np.asarray(self._image(obs, k)) for k in self.image_keys],
            "lang": instruction,
        }
        if self.send_state:
            example["state"] = self.build_state(obs)
        query = {"examples": [example], "do_sample": False, "unnorm_key": self.unnorm_key}
        if self.num_steps is not None:
            query["num_steps"] = self.num_steps
        response = self.client.predict_action(query)
        if response.get("status") != "ok":
            raise RuntimeError(f"policy server error: {response.get('error')}")
        actions = np.asarray(response["data"]["actions"])[0]  # (chunk, 12), already un-normalized
        return actions[: self.n_action_steps]

    def close(self) -> None:
        self.client.close()
