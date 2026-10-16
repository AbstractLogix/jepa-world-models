"""Synthetic MountainCar rendering and seeded data collection for study v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from jepa_tutorial.decision_study import (
    GOAL_POSITION,
    MAX_EPISODE_STEPS,
    MAX_POSITION,
    MAX_SPEED,
    MIN_POSITION,
)

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 96
RESIZED_WIDTH = 128
RESIZED_HEIGHT = 48
X_MIN_PIXEL = 8
X_MAX_PIXEL = IMAGE_WIDTH - 8


@dataclass(frozen=True)
class TransitionData:
    frames: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    next_states: np.ndarray
    episode_ids: np.ndarray


@dataclass(frozen=True)
class VisualStart:
    start_id: str
    seed: int
    step: int
    state: np.ndarray
    previous_frame: np.ndarray
    current_frame: np.ndarray


_PIXELS_X, _PIXELS_Y = np.meshgrid(
    np.arange(IMAGE_WIDTH, dtype=np.float64),
    np.arange(IMAGE_HEIGHT, dtype=np.float64),
)


def position_to_pixel(position: float) -> float:
    fraction = (float(position) - MIN_POSITION) / (MAX_POSITION - MIN_POSITION)
    return X_MIN_PIXEL + np.clip(fraction, 0.0, 1.0) * (X_MAX_PIXEL - X_MIN_PIXEL)


def pixel_to_position(pixel: float, image_width: int = IMAGE_WIDTH) -> float:
    left = X_MIN_PIXEL * image_width / IMAGE_WIDTH
    right = X_MAX_PIXEL * image_width / IMAGE_WIDTH
    fraction = (float(pixel) - left) / (right - left)
    return MIN_POSITION + np.clip(fraction, 0.0, 1.0) * (MAX_POSITION - MIN_POSITION)


def track_height(position: float) -> int:
    return int(70 - 20 * np.sin(3.0 * float(position)))


def _make_background() -> np.ndarray:
    background = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (18, 24, 34))
    draw = ImageDraw.Draw(background)
    points = [
        (x, track_height(pixel_to_position(x)))
        for x in range(X_MIN_PIXEL, X_MAX_PIXEL + 1, 2)
    ]
    draw.line(points, fill=(150, 160, 170), width=2)
    return np.asarray(background, dtype=np.uint8)


_STATIC_BACKGROUND = _make_background()


def render_frame(position: float) -> np.ndarray:
    """Render the same fixed red-car/green-goal image used by Notebook 22."""
    frame = _STATIC_BACKGROUND.copy()
    car_x = position_to_pixel(position)
    car_y = track_height(position) - 7
    goal_x = position_to_pixel(GOAL_POSITION)
    goal_y = track_height(GOAL_POSITION) - 18
    car_blob = np.exp(
        -0.5 * (((_PIXELS_X - car_x) / 1.7) ** 2 + ((_PIXELS_Y - car_y) / 1.7) ** 2)
    )
    goal_blob = np.exp(
        -0.5 * (((_PIXELS_X - goal_x) / 1.7) ** 2 + ((_PIXELS_Y - goal_y) / 1.7) ** 2)
    )
    frame[..., 0] = np.clip(frame[..., 0] + 230 * car_blob, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(frame[..., 1] + 230 * goal_blob, 0, 255).astype(np.uint8)
    return frame


def resize_frame(frame: np.ndarray) -> np.ndarray:
    resized = np.asarray(
        Image.fromarray(frame).resize(
            (RESIZED_WIDTH, RESIZED_HEIGHT), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    return resized


def color_center_x(frame: np.ndarray, color: str) -> float:
    red, green, blue = (frame[..., index].astype(np.float64) for index in range(3))
    if color == "red":
        weights = np.maximum(red - np.maximum(green, blue), 0.0)
    elif color == "green":
        weights = np.maximum(green - np.maximum(red, blue), 0.0)
    else:
        raise ValueError("color must be red or green")
    column_weights = weights.sum(axis=0)
    total = float(column_weights.sum())
    if total <= 0.0:
        raise ValueError(f"No {color} marker in rendered frame")
    return float(np.dot(np.arange(frame.shape[1]), column_weights) / total)


def visual_state(previous_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
    """Handcrafted position and two-frame velocity estimate for the direct policy."""
    position = pixel_to_position(color_center_x(current_frame, "red"), current_frame.shape[1])
    previous_position = pixel_to_position(
        color_center_x(previous_frame, "red"), previous_frame.shape[1]
    )
    velocity = float(np.clip(position - previous_position, -MAX_SPEED, MAX_SPEED))
    return np.asarray([position, velocity], dtype=np.float32)


def context_array(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    if previous.shape != current.shape or previous.shape != (RESIZED_HEIGHT, RESIZED_WIDTH, 3):
        raise ValueError("frames must both have shape [48, 128, 3]")
    return np.concatenate((previous, current), axis=-1)


def make_env() -> gym.Env:
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=MAX_EPISODE_STEPS)
    env.unwrapped.goal_position = GOAL_POSITION
    return env


def collect_random_transitions(data_seed: int, episodes: int = 20) -> TransitionData:
    """Collect random-action transitions with a stable episode-level identity."""
    if episodes < 2:
        raise ValueError("episodes must be at least 2")
    frames: list[np.ndarray] = []
    actions: list[float] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    env = make_env()
    try:
        for episode_id in range(episodes):
            episode_seed = data_seed * 100 + episode_id
            observation, _ = env.reset(seed=episode_seed)
            previous_frame = resize_frame(render_frame(float(observation[0])))
            current_frame = previous_frame.copy()
            rng = np.random.default_rng(episode_seed + 50_000)
            for _step in range(MAX_EPISODE_STEPS):
                state = np.asarray(env.unwrapped.state, dtype=np.float32).copy()
                action = float(rng.uniform(-1.0, 1.0))
                next_observation, _reward, terminated, truncated, _info = env.step(
                    np.asarray([action], dtype=np.float32)
                )
                next_state = np.asarray(env.unwrapped.state, dtype=np.float32).copy()
                next_frame = resize_frame(render_frame(float(next_observation[0])))
                frames.append(np.stack((previous_frame, current_frame, next_frame)))
                actions.append(action)
                states.append(state)
                next_states.append(next_state)
                episode_ids.append(episode_id)
                previous_frame, current_frame = current_frame, next_frame
                if terminated or truncated:
                    break
    finally:
        env.close()
    return TransitionData(
        frames=np.stack(frames).astype(np.uint8, copy=False),
        actions=np.asarray(actions, dtype=np.float32),
        states=np.stack(states).astype(np.float32),
        next_states=np.stack(next_states).astype(np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int16),
    )


def collect_intervention_starts(
    seeds: Sequence[int], steps: Sequence[int] = (40, 90, 140, 190)
) -> list[VisualStart]:
    """Collect fixed states along separate random-action episodes."""
    starts: list[VisualStart] = []
    wanted_steps = set(int(step) for step in steps)
    if not wanted_steps or min(wanted_steps) < 0:
        raise ValueError("steps must contain nonnegative values")
    env = make_env()
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            previous_frame = resize_frame(render_frame(float(observation[0])))
            current_frame = previous_frame.copy()
            rng = np.random.default_rng(seed + 70_000)
            for step in range(max(wanted_steps) + 1):
                if step in wanted_steps:
                    starts.append(
                        VisualStart(
                            start_id=f"{seed}-{step}",
                            seed=int(seed),
                            step=step,
                            state=np.asarray(env.unwrapped.state, dtype=np.float32).copy(),
                            previous_frame=previous_frame.copy(),
                            current_frame=current_frame.copy(),
                        )
                    )
                action = float(rng.uniform(-1.0, 1.0))
                next_observation, _reward, terminated, truncated, _info = env.step(
                    np.asarray([action], dtype=np.float32)
                )
                next_frame = resize_frame(render_frame(float(next_observation[0])))
                previous_frame, current_frame = current_frame, next_frame
                if terminated or truncated:
                    break
    finally:
        env.close()
    expected_count = len(seeds) * len(wanted_steps)
    if len(starts) != expected_count:
        raise RuntimeError(f"expected {expected_count} intervention starts, got {len(starts)}")
    return starts
