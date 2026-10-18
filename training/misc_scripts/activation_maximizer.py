"""Visualize input patterns that maximize the classifier's output classes.

Run this module from the repository root, for example::

    PYTHONPATH=. venv/bin/python training/misc_scripts/activation_maximizer.py \
        --symbol latex2e-_alpha
    PYTHONPATH=. venv/bin/python training/misc_scripts/activation_maximizer.py --all

The raw branch optimizes pixels directly. The stroke branch starts from one sample
per class produced by the training data loader, optimizes its polyline points through
a differentiable renderer, and retains only improvements verified by the production
Pillow rasterizer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtSvg as Qs
import PySide6.QtWidgets as Qw
import torch
from safetensors.torch import load_file
from sklearn.preprocessing import LabelEncoder
from torch import nn

import handtex.symbol_relations as sr
import handtex.utils as ut
import training.database
import training.data_loader as dl
from handtex.detector import image_gen as ig
from handtex.detector.image_gen import IMAGE_SIZE
from handtex.detector.inference import load_decoder
from handtex.detector.model import CNN
from handtex.worker_thread import Abort, Worker


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Parameters shared by the raw-pixel and stroke optimizations."""

    seed: int = 11
    steps: int = 100
    learning_rate: float = 0.04
    noise_standard_deviation: float = 0.08
    batch_size: int = 8
    image_size: int = IMAGE_SIZE
    points_per_stroke: int = 24
    anchor_weight: float = 20.0
    shape_weight: float = 0.5
    exact_interval: int = 10
    exact_proposals: int = 8

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("The random seed must be non-negative.")
        if self.steps < 1:
            raise ValueError("Optimization steps must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("The learning rate must be greater than zero.")
        if self.noise_standard_deviation < 0:
            raise ValueError("The noise standard deviation cannot be negative.")
        if self.batch_size < 1:
            raise ValueError("The batch size must be at least 1.")
        if self.image_size < 2:
            raise ValueError("The image size must be at least 2 pixels.")
        if self.points_per_stroke < 2:
            raise ValueError("Each stroke must contain at least two points.")
        if self.anchor_weight < 0 or self.shape_weight < 0:
            raise ValueError("Stroke regularization weights cannot be negative.")
        if self.exact_interval < 1:
            raise ValueError("The exact-raster interval must be at least 1.")
        if self.exact_proposals < 0:
            raise ValueError("The number of exact-raster proposals cannot be negative.")


@dataclass(frozen=True, slots=True)
class ActivationResult:
    """The two optimized inputs and final target scores for one output class."""

    class_index: int
    symbol_key: str
    raw_image: np.ndarray
    stroke_image: np.ndarray
    raw_logit: float
    raw_probability: float
    stroke_logit: float
    stroke_probability: float
    stroke_count: int
    initial_stroke_probability: float
    initial_stroke_logit: float


@dataclass(slots=True)
class ActivationEntry:
    """Mutable display state for one list row."""

    class_index: int
    symbol_key: str
    status: str = "Queued"
    result: ActivationResult | None = None


def resample_polyline(points: Sequence[Sequence[float]], point_count: int) -> np.ndarray:
    """Resample a polyline to evenly spaced points along its arc length."""

    polyline = np.asarray(points, dtype=np.float32)
    if polyline.ndim != 2 or polyline.shape[1] != 2 or len(polyline) == 0:
        raise ValueError("A stroke must contain one or more 2D points.")
    if point_count < 2:
        raise ValueError("A resampled stroke needs at least two points.")
    if len(polyline) == 1:
        return np.repeat(polyline, point_count, axis=0)

    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] == 0:
        return np.repeat(polyline[:1], point_count, axis=0)
    positions = np.linspace(0, cumulative[-1], point_count)
    return np.stack(
        [
            np.interp(positions, cumulative, polyline[:, 0]),
            np.interp(positions, cumulative, polyline[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)


def resample_drawing(
    strokes: Sequence[Sequence[Sequence[float]]], point_count: int, image_size: int
) -> np.ndarray:
    """Resample normalized 1000-space strokes into model-image coordinates."""

    if not strokes:
        raise ValueError("A drawing must contain at least one stroke.")
    padding = 0.10
    scale = image_size * (1 - padding) / 1000
    offset = image_size * padding / 2
    sampled = [resample_polyline(stroke, point_count) * scale + offset for stroke in strokes]
    return np.stack(sampled).astype(np.float32)


def render_soft_strokes(
    points: torch.Tensor,
    active_strokes: torch.Tensor,
    *,
    image_size: int,
    softness: float = 0.25,
) -> torch.Tensor:
    """Render differentiable one-pixel polylines as normalized NCHW images."""

    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError("Expected stroke points with shape [batch, strokes, points, 2].")
    if active_strokes.shape != points.shape[:2]:
        raise ValueError("The active-stroke mask does not match the point tensor.")
    if softness <= 0:
        raise ValueError("Stroke softness must be greater than zero.")

    y_coordinates, x_coordinates = torch.meshgrid(
        torch.arange(image_size, dtype=points.dtype, device=points.device),
        torch.arange(image_size, dtype=points.dtype, device=points.device),
        indexing="ij",
    )
    pixel_grid = torch.stack((x_coordinates, y_coordinates), dim=-1)[None, None, None]
    starts = points[:, :, :-1, None, None, :]
    ends = points[:, :, 1:, None, None, :]
    segment_vectors = ends - starts
    from_start = pixel_grid - starts
    projection = (from_start * segment_vectors).sum(dim=-1) / (
        segment_vectors.square().sum(dim=-1) + 1e-6
    )
    projection = projection.clamp(0.0, 1.0)
    closest = starts + projection[..., None] * segment_vectors
    distances = (pixel_grid - closest).square().sum(dim=-1).add(1e-8).sqrt()
    segment_ink = torch.sigmoid((0.35 - distances) / softness)
    segment_ink = segment_ink * active_strokes[:, :, None, None, None]
    transmittance = torch.prod((1.0 - segment_ink).flatten(1, 2), dim=1)
    ink = 1.0 - transmittance
    return 1.0 - 2.0 * ink[:, None]


def rasterize_stroke_points(
    points: np.ndarray, stroke_counts: Sequence[int], image_size: int
) -> np.ndarray:
    """Render optimized model-space points through the production Pillow path."""

    padding = 0.10
    scale = image_size * (1 - padding) / 1000
    offset = image_size * padding / 2
    images = []
    for drawing, stroke_count in zip(points, stroke_counts):
        strokes = [
            [((float(x) - offset) / scale, (float(y) - offset) / scale) for x, y in stroke]
            for stroke in drawing[:stroke_count]
        ]
        images.append(ig.strokes_to_grayscale_image(strokes, image_size))
    return np.stack(images).astype(np.uint8)


def _pad_stroke_points(drawings: Sequence[np.ndarray]) -> np.ndarray:
    """Pad drawings to a common stroke count for batched optimization."""

    if not drawings:
        raise ValueError("At least one stroke drawing is required.")
    maximum_strokes = max(len(drawing) for drawing in drawings)
    point_count = drawings[0].shape[1]
    padded = np.zeros((len(drawings), maximum_strokes, point_count, 2), dtype=np.float32)
    for row, drawing in enumerate(drawings):
        if drawing.shape[1:] != (point_count, 2):
            raise ValueError("All stroke drawings must use the same point count.")
        padded[row, : len(drawing)] = drawing
    return padded


def build_seed_stroke_dataset(
    database_path: Path,
    symbol_data: sr.SymbolData,
    image_size: int,
    random_seed: int,
) -> dl.StrokeDataset:
    """Build the training pipeline's first available stroke sample per class.

    ``StrokeDataset`` resolves similarity, symmetry, negation, and inside-composition
    paths before the sample is returned. Consequently, model classes with no direct
    database rows still receive the same derived stroke representation used in
    training.
    """

    if not database_path.is_file():
        raise ValueError(f"Stroke database does not exist: {database_path}")
    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_data.leaders)
    split_percentages = {
        dl.DataSplit.TRAIN: 80,
        dl.DataSplit.VALIDATION: 1,
        dl.DataSplit.TEST: 19,
    }
    dataset = dl.StrokeDataset(
        database_path,
        symbol_data,
        image_size,
        label_encoder,
        random_seed,
        split_percentages,
        split=dl.DataSplit.TEST,
        class_limit=1,
        random_augmentation=False,
        debug_single_sample_only=True,
    )
    missing = [key for key in symbol_data.leaders if key not in dataset.symbol_keys]
    if missing:
        raise ValueError(f"The training data loader could not derive a sample for {missing[0]!r}.")
    return dataset


def initialize_inputs(
    class_indices: Sequence[int], config: ActivationConfig, device: torch.device | str
) -> torch.Tensor:
    """Build deterministic, nearly-white starting images for output classes.

    Each class owns its random stream, so its starting image is unchanged by
    batch size, queue order, or whether it was opened alone or through ``--all``.
    Noise is generated on the CPU to also keep CPU/CUDA starts identical.
    """

    config.validate()
    starts = []
    maximum_seed = 2**63 - 1
    for class_index in class_indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((config.seed + int(class_index)) % maximum_seed)
        noise = torch.randn(
            (1, config.image_size, config.image_size), generator=generator, dtype=torch.float32
        )
        starts.append(
            (torch.ones_like(noise) + config.noise_standard_deviation * noise).clamp(-1.0, 1.0)
        )
    if not starts:
        return torch.empty((0, 1, config.image_size, config.image_size), device=device)
    return torch.stack(starts).to(device)


def _abort_requested(abort_flag: object | None) -> bool:
    return abort_flag is not None and bool(abort_flag.get())


def _to_grayscale_uint8(images: torch.Tensor) -> np.ndarray:
    pixels = ((images.detach().cpu() + 1.0) * 127.5).round().clamp(0, 255)
    return np.ascontiguousarray(pixels[:, 0].to(torch.uint8).numpy())


def _normalized_exact_images(images: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(images).to(device=device, dtype=torch.float32)[:, None] / 127.5 - 1.0


def generate_stroke_activation_batch(
    model: nn.Module,
    class_indices: Sequence[int],
    encodings: Sequence[str],
    config: ActivationConfig,
    *,
    stroke_dataset: dl.StrokeDataset,
    device: torch.device | str | None = None,
    abort_flag: object | None = None,
) -> list[ActivationResult]:
    """Optimize human-style polyline points and retain the best exact raster."""

    config.validate()
    class_indices = [int(index) for index in class_indices]
    if not class_indices:
        return []
    if min(class_indices) < 0 or max(class_indices) >= len(encodings):
        raise IndexError("A requested class index is outside the encoding list.")
    if stroke_dataset.image_size != config.image_size:
        raise ValueError(
            f"Stroke dataset images are {stroke_dataset.image_size}px, but activation images "
            f"are {config.image_size}px."
        )
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selected_strokes = []
    selected_points = []
    for class_index in class_indices:
        if _abort_requested(abort_flag):
            raise Abort()
        symbol_key = encodings[class_index]
        sample_range = stroke_dataset.range_for_symbol(symbol_key)
        if len(sample_range) != 1:
            raise ValueError(
                f"Expected one training-loader sample for {symbol_key!r}, "
                f"but found {len(sample_range)}."
            )
        strokes, loaded_key = stroke_dataset.load_transformed_strokes(sample_range.start)
        if loaded_key != symbol_key:
            raise ValueError(
                f"Training loader returned {loaded_key!r} while loading {symbol_key!r}."
            )
        selected_strokes.append(strokes)
        selected_points.append(
            resample_drawing(strokes, config.points_per_stroke, config.image_size)
        )

    stroke_counts = [len(points) for points in selected_points]
    padded_points = _pad_stroke_points(selected_points)
    # Preserve the loader sample byte-for-byte as the baseline. Resampling is
    # necessary to give the optimizer fixed-size point tensors, but can otherwise
    # move a sharp corner by a pixel before optimization has even begun.
    seed_images = np.stack(
        [ig.strokes_to_grayscale_image(strokes, config.image_size) for strokes in selected_strokes]
    ).astype(np.uint8)
    with torch.no_grad():
        seed_logits = model(_normalized_exact_images(seed_images, device))
        if seed_logits.ndim != 2 or seed_logits.shape[1] != len(encodings):
            raise ValueError(
                f"Model produced {tuple(seed_logits.shape)}, but {len(encodings)} output "
                "classes were expected."
            )
    maximum_strokes = padded_points.shape[1]
    active_strokes = np.zeros((len(class_indices), maximum_strokes), dtype=np.float32)
    for row, points in enumerate(selected_points):
        active_strokes[row, : len(points)] = 1

    anchor_points = torch.from_numpy(padded_points).to(device)
    stroke_points = anchor_points.clone().requires_grad_(True)
    active_tensor = torch.from_numpy(active_strokes).to(device)
    target_indices = torch.tensor(class_indices, dtype=torch.long, device=device)
    batch_rows = torch.arange(len(class_indices), device=device)
    raw_images = initialize_inputs(class_indices, config, device).requires_grad_(True)
    optimizer = torch.optim.Adam([raw_images, stroke_points], lr=config.learning_rate)

    best_images = seed_images.copy()
    best_points = padded_points.copy()
    best_logits = seed_logits[batch_rows, target_indices].detach().clone()
    best_objectives = best_logits.clone()
    best_probabilities = torch.softmax(seed_logits, dim=1)[batch_rows, target_indices].detach()
    initial_probabilities = best_probabilities.detach().cpu().numpy().copy()
    initial_logits = best_logits.detach().cpu().numpy().copy()

    for step in range(config.steps):
        if _abort_requested(abort_flag):
            raise Abort()
        optimizer.zero_grad(set_to_none=True)
        softness = 0.35 - 0.10 * step / max(1, config.steps - 1)
        stroke_images = render_soft_strokes(
            stroke_points,
            active_tensor,
            image_size=config.image_size,
            softness=softness,
        )
        logits = model(torch.cat((raw_images, stroke_images)))
        raw_logits = logits[: len(class_indices)]
        stroke_logits = logits[len(class_indices) :]
        raw_objective = raw_logits[batch_rows, target_indices]
        stroke_target_logits = stroke_logits[batch_rows, target_indices]

        point_mask = active_tensor[:, :, None, None]
        point_denominator = point_mask.sum(dim=(1, 2, 3)).clamp_min(1) * (
            config.points_per_stroke * 2
        )
        anchor_error = (
            ((stroke_points - anchor_points) / config.image_size).square() * point_mask
        ).sum(dim=(1, 2, 3)) / point_denominator
        current_vectors = stroke_points[:, :, 1:] - stroke_points[:, :, :-1]
        anchor_vectors = anchor_points[:, :, 1:] - anchor_points[:, :, :-1]
        vector_mask = active_tensor[:, :, None, None]
        vector_denominator = vector_mask.sum(dim=(1, 2, 3)).clamp_min(1) * (
            (config.points_per_stroke - 1) * 2
        )
        shape_error = (
            ((current_vectors - anchor_vectors) / config.image_size).square() * vector_mask
        ).sum(dim=(1, 2, 3)) / vector_denominator
        stroke_objective = (
            stroke_target_logits
            - config.anchor_weight * anchor_error
            - config.shape_weight * shape_error
        )
        loss = -raw_objective.mean() - stroke_objective.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            raw_images.clamp_(-1.0, 1.0)
            margin = config.image_size * 0.05
            stroke_points.clamp_(margin, config.image_size - margin)

        score_exact = (step + 1) % config.exact_interval == 0 or step + 1 == config.steps
        if score_exact:
            current_points = stroke_points.detach().cpu().numpy()
            proposal_points = []
            proposal_targets = []
            proposal_ranges = []
            for row, (class_index, stroke_count) in enumerate(zip(class_indices, stroke_counts)):
                start = len(proposal_points)
                proposal_points.append(current_points[row].copy())
                proposal_targets.append(class_index)
                generator = np.random.default_rng(
                    config.seed + class_index * 1009 + (step + 1) * 9176
                )
                for _ in range(config.exact_proposals):
                    proposal = best_points[row].copy()
                    stroke_index = int(generator.integers(stroke_count))
                    point_index = int(generator.integers(config.points_per_stroke))
                    axis = int(generator.integers(2))
                    delta = -1.0 if generator.integers(2) == 0 else 1.0
                    proposal[stroke_index, point_index, axis] += delta
                    proposal[stroke_index, point_index, axis] = np.clip(
                        proposal[stroke_index, point_index, axis],
                        config.image_size * 0.05,
                        config.image_size * 0.95,
                    )
                    proposal_points.append(proposal)
                    proposal_targets.append(class_index)
                proposal_ranges.append(range(start, len(proposal_points)))

            proposal_counts = [
                stroke_counts[row]
                for row in range(len(class_indices))
                for _ in range(1 + config.exact_proposals)
            ]
            exact_images = rasterize_stroke_points(
                np.stack(proposal_points), proposal_counts, config.image_size
            )
            with torch.no_grad():
                exact_logits = model(_normalized_exact_images(exact_images, device))
                exact_probabilities = torch.softmax(exact_logits, dim=1)
                for row, proposal_range in enumerate(proposal_ranges):
                    indices = list(proposal_range)
                    class_index = class_indices[row]
                    proposed_logits = exact_logits[indices, class_index]
                    proposed_probabilities = exact_probabilities[indices, class_index]
                    local_points = np.stack(
                        proposal_points[proposal_range.start : proposal_range.stop]
                    )[:, : stroke_counts[row]]
                    local_anchor = padded_points[row, : stroke_counts[row]]
                    anchor_errors = np.mean(
                        ((local_points - local_anchor[None]) / config.image_size) ** 2,
                        axis=(1, 2, 3),
                    )
                    local_vectors = np.diff(local_points, axis=2)
                    anchor_vectors = np.diff(local_anchor, axis=1)
                    shape_errors = np.mean(
                        ((local_vectors - anchor_vectors[None]) / config.image_size) ** 2,
                        axis=(1, 2, 3),
                    )
                    proposed_objectives = proposed_logits - torch.from_numpy(
                        config.anchor_weight * anchor_errors + config.shape_weight * shape_errors
                    ).to(device)
                    valid = (proposed_probabilities >= best_probabilities[row]) & (
                        proposed_logits > best_logits[row]
                    )
                    if not valid.any():
                        continue
                    valid_objectives = proposed_objectives.masked_fill(~valid, -torch.inf)
                    local_index = int(valid_objectives.argmax().item())
                    if proposed_objectives[local_index] <= best_objectives[row]:
                        continue
                    candidate_index = proposal_range.start + local_index
                    best_images[row] = exact_images[candidate_index]
                    best_points[row] = proposal_points[candidate_index]
                    best_logits[row] = proposed_logits[local_index]
                    best_probabilities[row] = proposed_probabilities[local_index]
                    best_objectives[row] = proposed_objectives[local_index]

    if _abort_requested(abort_flag):
        raise Abort()
    with torch.no_grad():
        final_raw_logits = model(raw_images)
        final_raw_probabilities = torch.softmax(final_raw_logits, dim=1)

    results = []
    raw_uint8 = _to_grayscale_uint8(raw_images)
    for row, class_index in enumerate(class_indices):
        results.append(
            ActivationResult(
                class_index=class_index,
                symbol_key=encodings[class_index],
                raw_image=raw_uint8[row],
                stroke_image=best_images[row],
                raw_logit=float(final_raw_logits[row, class_index].item()),
                raw_probability=float(final_raw_probabilities[row, class_index].item()),
                stroke_logit=float(best_logits[row].item()),
                stroke_probability=float(best_probabilities[row].item()),
                stroke_count=stroke_counts[row],
                initial_stroke_probability=float(initial_probabilities[row]),
                initial_stroke_logit=float(initial_logits[row]),
            )
        )
    return results


def resolve_symbol_key(
    symbol_key: str, symbol_data: sr.SymbolData, encodings: Sequence[str]
) -> tuple[str, int]:
    """Resolve any metadata symbol to the leader represented by the model."""

    if symbol_key not in symbol_data:
        suggestion = ut.closest_match(symbol_key, symbol_data.all_keys)
        suffix = f" Did you mean {suggestion!r}?" if suggestion else ""
        raise ValueError(f"Unknown symbol key {symbol_key!r}.{suffix}")
    leader = symbol_data.to_leader.get(symbol_key, symbol_key)
    try:
        return leader, encodings.index(leader)
    except ValueError as error:
        raise ValueError(
            f"The equivalence-class leader {leader!r} is absent from the model encodings."
        ) from error


def select_next_rows(
    pending_rows: set[int], visible_rows: Sequence[int], batch_size: int
) -> list[int]:
    """Select visible pending rows first, then continue in encoding order."""

    if batch_size < 1:
        raise ValueError("The batch size must be at least 1.")
    selected = []
    seen = set()
    for row in visible_rows:
        if row in pending_rows and row not in seen:
            selected.append(row)
            seen.add(row)
            if len(selected) == batch_size:
                return selected
    for row in sorted(pending_rows):
        if row not in seen:
            selected.append(row)
            if len(selected) == batch_size:
                break
    return selected


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but it is not available to PyTorch.")
    return torch.device(requested)


def load_visualization_model(
    model_path: Path, encodings_path: Path, device: torch.device
) -> tuple[nn.Module, list[str]]:
    """Load either a safetensors model or a training checkpoint."""

    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    if not encodings_path.is_file():
        raise ValueError(f"Encoding file does not exist: {encodings_path}")

    decoder = load_decoder(encodings_path)
    encodings = [decoder[index] for index in range(len(decoder))]
    model = CNN(num_classes=len(encodings), image_size=IMAGE_SIZE)

    if model_path.suffix == ".safetensors":
        state_dict = load_file(str(model_path), device="cpu")
    else:
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch versions.
            checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"No model state dictionary was found in {model_path}.")

    output_weights = state_dict.get("fc1.weight")
    if output_weights is not None and output_weights.shape[0] != len(encodings):
        raise ValueError(
            f"The model has {output_weights.shape[0]} output classes, but the encoding file "
            f"contains {len(encodings)} symbols."
        )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "The model state is not compatible with the current Hand TeX CNN architecture."
        ) from error
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, encodings


class ActivationListModel(Qc.QAbstractListModel):
    EntryRole = Qc.Qt.ItemDataRole.UserRole + 1

    def __init__(self, entries: list[ActivationEntry], parent: Qc.QObject | None = None):
        super().__init__(parent)
        self.entries = entries
        self.row_by_class_index = {entry.class_index: row for row, entry in enumerate(self.entries)}

    def rowCount(self, parent: Qc.QModelIndex = Qc.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.entries)

    def data(self, index: Qc.QModelIndex, role: int = Qc.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.entries):
            return None
        entry = self.entries[index.row()]
        if role == Qc.Qt.ItemDataRole.DisplayRole:
            return entry.symbol_key
        if role == Qc.Qt.ItemDataRole.ToolTipRole:
            return f"Class {entry.class_index}: {entry.symbol_key}"
        if role == self.EntryRole:
            return entry
        return None

    def reset_results(self) -> None:
        for entry in self.entries:
            entry.status = "Queued"
            entry.result = None
        if self.entries:
            self.dataChanged.emit(self.index(0), self.index(len(self.entries) - 1))

    def set_status(self, rows: Sequence[int], status: str) -> None:
        for row in rows:
            self.entries[row].status = status
            self.dataChanged.emit(self.index(row), self.index(row), [self.EntryRole])

    def set_result(self, result: ActivationResult) -> int | None:
        row = self.row_by_class_index.get(result.class_index)
        if row is None:
            return None
        entry = self.entries[row]
        entry.result = result
        entry.status = "Complete"
        self.dataChanged.emit(self.index(row), self.index(row), [self.EntryRole])
        return row


class ActivationDelegate(Qw.QStyledItemDelegate):
    CARD_SIZE = Qc.QSize(390, 205)

    def __init__(self, symbol_data: sr.SymbolData, parent: Qc.QObject | None = None):
        super().__init__(parent)
        self.symbol_data = symbol_data
        self._svg_cache: dict[tuple[str, str], Qg.QPixmap] = {}

    def sizeHint(self, option: Qw.QStyleOptionViewItem, index: Qc.QModelIndex) -> Qc.QSize:
        return self.CARD_SIZE

    def _symbol_pixmap(self, key: str, color: str) -> Qg.QPixmap:
        cache_key = (key, color)
        if cache_key not in self._svg_cache:
            pixmap = Qg.QPixmap(96, 96)
            pixmap.fill(Qc.Qt.GlobalColor.transparent)
            renderer = Qs.QSvgRenderer(ut.load_symbol_svg(self.symbol_data[key], color))
            renderer.setAspectRatioMode(Qc.Qt.AspectRatioMode.KeepAspectRatio)
            painter = Qg.QPainter(pixmap)
            renderer.render(painter, Qc.QRectF(0, 0, 96, 96))
            painter.end()
            self._svg_cache[cache_key] = pixmap
        return self._svg_cache[cache_key]

    @staticmethod
    def _draw_activation(
        painter: Qg.QPainter,
        image: np.ndarray,
        target: Qc.QRect,
    ) -> None:
        height, width = image.shape
        qt_image = Qg.QImage(
            image.tobytes(),
            width,
            height,
            width,
            Qg.QImage.Format.Format_Grayscale8,
        )
        painter.drawImage(target, qt_image)

    def paint(
        self,
        painter: Qg.QPainter,
        option: Qw.QStyleOptionViewItem,
        index: Qc.QModelIndex,
    ) -> None:
        entry: ActivationEntry = index.data(ActivationListModel.EntryRole)
        if entry is None:
            return

        painter.save()
        painter.setRenderHints(
            Qg.QPainter.RenderHint.Antialiasing | Qg.QPainter.RenderHint.SmoothPixmapTransform
        )
        rect = option.rect.adjusted(4, 4, -4, -4)
        palette = option.palette
        selected = bool(option.state & Qw.QStyle.StateFlag.State_Selected)
        background = palette.color(
            Qg.QPalette.ColorRole.Highlight if selected else Qg.QPalette.ColorRole.Base
        )
        foreground = palette.color(
            Qg.QPalette.ColorRole.HighlightedText if selected else Qg.QPalette.ColorRole.Text
        )
        painter.setPen(palette.color(Qg.QPalette.ColorRole.Mid))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(foreground)

        leader_rect = Qc.QRect(rect.left() + 10, rect.top() + 55, 82, 82)
        painter.drawPixmap(leader_rect, self._symbol_pixmap(entry.symbol_key, foreground.name()))
        bold_font = painter.font()
        bold_font.setBold(True)
        painter.setFont(bold_font)
        key_rect = Qc.QRect(rect.left() + 10, rect.top() + 8, rect.width() - 20, 28)
        painter.drawText(
            key_rect,
            Qc.Qt.AlignmentFlag.AlignLeft | Qc.Qt.AlignmentFlag.AlignVCenter,
            entry.symbol_key,
        )
        painter.setFont(option.font)
        columns = (("Raw pixels", rect.left() + 98), ("Strokes", rect.left() + 237))
        for label, left in columns:
            painter.drawText(
                Qc.QRect(left, rect.top() + 32, 132, 22),
                Qc.Qt.AlignmentFlag.AlignCenter,
                label,
            )

        result = entry.result
        if result is None:
            for _, left in columns:
                placeholder = Qc.QRect(left + 21, rect.top() + 55, 90, 90)
                painter.setPen(palette.color(Qg.QPalette.ColorRole.Mid))
                painter.drawRect(placeholder)
                painter.drawText(placeholder, Qc.Qt.AlignmentFlag.AlignCenter, entry.status)
            painter.restore()
            return

        raw_rect = Qc.QRect(columns[0][1] + 21, rect.top() + 55, 90, 90)
        stroke_rect = Qc.QRect(columns[1][1] + 21, rect.top() + 55, 90, 90)
        self._draw_activation(painter, result.raw_image, raw_rect)
        self._draw_activation(painter, result.stroke_image, stroke_rect)
        painter.setPen(foreground)
        painter.drawText(
            Qc.QRect(columns[0][1], rect.top() + 150, 132, 24),
            Qc.Qt.AlignmentFlag.AlignCenter,
            f"logit {result.raw_logit:.2f}  p {result.raw_probability:.0%}",
        )
        painter.drawText(
            Qc.QRect(columns[1][1], rect.top() + 150, 132, 24),
            Qc.Qt.AlignmentFlag.AlignCenter,
            f"logit {result.stroke_logit:.2f}  p {result.stroke_probability:.0%}",
        )
        if result.stroke_count is not None:
            painter.setPen(foreground)
            details = (
                f"{result.stroke_count} {ut.f_plural(result.stroke_count, 'stroke')}  "
                f"Δlogit {result.stroke_logit - result.initial_stroke_logit:+.1f}"
            )
            painter.drawText(
                Qc.QRect(columns[1][1], rect.top() + 174, 132, 20),
                Qc.Qt.AlignmentFlag.AlignCenter,
                details,
            )
        painter.restore()


class ActivationWindow(Qw.QWidget):
    abort_requested = Qc.Signal()

    def __init__(
        self,
        classifier: nn.Module,
        encodings: list[str],
        symbol_data: sr.SymbolData,
        class_indices: Sequence[int],
        config: ActivationConfig,
        device: torch.device,
        stroke_dataset: dl.StrokeDataset,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.encodings = encodings
        self.symbol_data = symbol_data
        self.device = device
        self.noise_standard_deviation = config.noise_standard_deviation
        self.image_size = config.image_size
        self.stroke_dataset = stroke_dataset
        self.generation_config = config
        self.generation = 0
        self.worker_running = False
        self.cancelled = False
        self.pending_rows: set[int] = set()
        self.in_flight_rows: set[int] = set()
        self.completed_rows: set[int] = set()
        # Keep the QRunnable and its signal owner alive until the pool reports
        # completion; QThreadPool does not retain the Python wrapper for us.
        self.active_worker: Worker | None = None
        self.thread_pool = Qc.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)

        entries = [
            ActivationEntry(class_index=index, symbol_key=encodings[index])
            for index in class_indices
        ]
        self.list_model = ActivationListModel(entries, self)
        self._build_ui(config)
        self.setWindowTitle("Hand TeX — Activation Maximizer")
        self.resize(1220, 820)
        Qc.QTimer.singleShot(0, self.regenerate)

    def _build_ui(self, config: ActivationConfig) -> None:
        outer = Qw.QVBoxLayout(self)
        controls = Qw.QGridLayout()

        self.seed_input = Qw.QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(config.seed)
        controls.addWidget(Qw.QLabel("Seed"), 0, 0)
        controls.addWidget(self.seed_input, 0, 1)

        self.steps_input = Qw.QSpinBox()
        self.steps_input.setRange(1, 5000)
        self.steps_input.setValue(config.steps)
        controls.addWidget(Qw.QLabel("Steps"), 0, 2)
        controls.addWidget(self.steps_input, 0, 3)

        self.learning_rate_input = Qw.QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setDecimals(4)
        self.learning_rate_input.setSingleStep(0.01)
        self.learning_rate_input.setValue(config.learning_rate)
        controls.addWidget(Qw.QLabel("Learning rate"), 0, 4)
        controls.addWidget(self.learning_rate_input, 0, 5)

        self.anchor_input = Qw.QDoubleSpinBox()
        self.anchor_input.setRange(0.0, 1000.0)
        self.anchor_input.setDecimals(2)
        self.anchor_input.setValue(config.anchor_weight)
        self.anchor_input.setToolTip("Keeps optimized points near the human seed drawing.")
        controls.addWidget(Qw.QLabel("Anchor weight"), 1, 0)
        controls.addWidget(self.anchor_input, 1, 1)

        self.shape_input = Qw.QDoubleSpinBox()
        self.shape_input.setRange(0.0, 1000.0)
        self.shape_input.setDecimals(2)
        self.shape_input.setValue(config.shape_weight)
        self.shape_input.setToolTip("Preserves the seed drawing's local stroke directions.")
        controls.addWidget(Qw.QLabel("Shape weight"), 1, 2)
        controls.addWidget(self.shape_input, 1, 3)

        self.points_input = Qw.QSpinBox()
        self.points_input.setRange(2, 128)
        self.points_input.setValue(config.points_per_stroke)
        controls.addWidget(Qw.QLabel("Points/stroke"), 1, 4)
        controls.addWidget(self.points_input, 1, 5)

        self.batch_input = Qw.QSpinBox()
        self.batch_input.setRange(1, 64)
        self.batch_input.setValue(config.batch_size)
        self.batch_input.setToolTip("Larger batches are faster but use more RAM or VRAM.")
        controls.addWidget(Qw.QLabel("Batch size"), 0, 6)
        controls.addWidget(self.batch_input, 0, 7)

        self.regenerate_button = Qw.QPushButton("Regenerate")
        self.regenerate_button.clicked.connect(self.regenerate)
        controls.addWidget(self.regenerate_button, 0, 8)
        self.cancel_button = Qw.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_generation)
        controls.addWidget(self.cancel_button, 1, 8)
        controls.setColumnStretch(9, 1)
        outer.addLayout(controls)

        progress_layout = Qw.QHBoxLayout()
        self.status_label = Qw.QLabel(f"Ready on {self.device}")
        progress_layout.addWidget(self.status_label)
        self.progress_bar = Qw.QProgressBar()
        self.progress_bar.setRange(0, len(self.list_model.entries))
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, 1)
        outer.addLayout(progress_layout)

        self.list_view = Qw.QListView()
        self.list_view.setModel(self.list_model)
        self.list_view.setItemDelegate(ActivationDelegate(self.symbol_data, self.list_view))
        self.list_view.setViewMode(Qw.QListView.ViewMode.IconMode)
        self.list_view.setFlow(Qw.QListView.Flow.LeftToRight)
        self.list_view.setWrapping(True)
        self.list_view.setResizeMode(Qw.QListView.ResizeMode.Adjust)
        self.list_view.setMovement(Qw.QListView.Movement.Static)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setGridSize(ActivationDelegate.CARD_SIZE)
        self.list_view.setSpacing(6)
        self.list_view.verticalScrollBar().valueChanged.connect(self._viewport_changed)
        outer.addWidget(self.list_view, 1)

        self.viewport_timer = Qc.QTimer(self)
        self.viewport_timer.setSingleShot(True)
        self.viewport_timer.setInterval(50)
        self.viewport_timer.timeout.connect(self.schedule_next_batch)

    def current_config(self) -> ActivationConfig:
        return ActivationConfig(
            seed=self.seed_input.value(),
            steps=self.steps_input.value(),
            learning_rate=self.learning_rate_input.value(),
            noise_standard_deviation=self.noise_standard_deviation,
            batch_size=self.batch_input.value(),
            image_size=self.image_size,
            points_per_stroke=self.points_input.value(),
            anchor_weight=self.anchor_input.value(),
            shape_weight=self.shape_input.value(),
            exact_interval=self.generation_config.exact_interval,
            exact_proposals=self.generation_config.exact_proposals,
        )

    def _viewport_changed(self) -> None:
        # If a batch is active, this timer records the latest viewport for the
        # selection made immediately after that batch completes.
        self.viewport_timer.start()

    def resizeEvent(self, event: Qg.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "viewport_timer"):
            self.viewport_timer.start()

    def visible_rows(self) -> list[int]:
        viewport_rect = self.list_view.viewport().rect()
        visible = []
        for row in range(self.list_model.rowCount()):
            index = self.list_model.index(row)
            if self.list_view.visualRect(index).intersects(viewport_rect):
                visible.append(row)
        return visible

    def regenerate(self) -> None:
        if self.worker_running:
            self.abort_requested.emit()
        self.generation += 1
        self.generation_config = self.current_config()
        self.cancelled = False
        self.pending_rows = set(range(self.list_model.rowCount()))
        self.in_flight_rows.clear()
        self.completed_rows.clear()
        self.list_model.reset_results()
        self.progress_bar.setRange(0, self.list_model.rowCount())
        self.progress_bar.setValue(0)
        class_word = ut.f_plural(len(self.pending_rows), "class")
        self.status_label.setText(f"Queued {len(self.pending_rows)} {class_word} on {self.device}")
        self.cancel_button.setEnabled(True)
        if not self.worker_running:
            Qc.QTimer.singleShot(0, self.schedule_next_batch)

    def cancel_generation(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        self.abort_requested.emit()
        unstarted = sorted(self.pending_rows)
        self.pending_rows.clear()
        self.list_model.set_status(unstarted, "Cancelled")
        self.status_label.setText(
            f"Cancelled — {len(self.completed_rows)}/{self.list_model.rowCount()} complete"
        )
        self.cancel_button.setEnabled(False)

    def schedule_next_batch(self) -> None:
        if self.worker_running or self.cancelled:
            return
        if not self.pending_rows:
            class_word = ut.f_plural(len(self.completed_rows), "class")
            self.status_label.setText(
                f"Complete — {len(self.completed_rows)} {class_word} generated on {self.device}"
            )
            self.cancel_button.setEnabled(False)
            return

        config = self.generation_config
        rows = select_next_rows(
            self.pending_rows, self.visible_rows(), min(config.batch_size, len(self.pending_rows))
        )
        self.pending_rows.difference_update(rows)
        self.in_flight_rows = set(rows)
        self.list_model.set_status(rows, "Optimizing…")
        class_indices = [self.list_model.entries[row].class_index for row in rows]
        generation = self.generation
        worker = Worker(
            generate_stroke_activation_batch,
            self.classifier,
            class_indices,
            self.encodings,
            config,
            device=self.device,
            stroke_dataset=self.stroke_dataset,
            abort_signal=self.abort_requested,
            no_progress_callback=True,
        )
        worker.signals.result.connect(partial(self._batch_result, generation))
        worker.signals.error.connect(partial(self._batch_error, generation))
        worker.signals.aborted.connect(partial(self._batch_aborted, generation))
        worker.signals.finished.connect(partial(self._batch_finished, generation))
        self.active_worker = worker
        self.worker_running = True
        self.status_label.setText(
            f"Optimizing {len(rows)} classes — "
            f"{len(self.completed_rows)}/{self.list_model.rowCount()} complete"
        )
        self.thread_pool.start(worker)

    def _batch_result(self, generation: int, results: list[ActivationResult]) -> None:
        if generation != self.generation:
            return
        for result in results:
            row = self.list_model.set_result(result)
            if row is not None:
                self.completed_rows.add(row)
        self.in_flight_rows.clear()
        self.progress_bar.setValue(len(self.completed_rows))

    def _batch_error(self, generation: int, error) -> None:
        if generation != self.generation:
            return
        failed_rows = sorted(self.in_flight_rows)
        self.list_model.set_status(failed_rows, "Error")
        self.in_flight_rows.clear()
        self.cancelled = True
        self.list_model.set_status(sorted(self.pending_rows), "Cancelled")
        self.pending_rows.clear()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Generation failed")
        Qw.QMessageBox.critical(self, "Activation Generation Failed", str(error))

    def _batch_aborted(self, generation: int, _details: tuple) -> None:
        if generation != self.generation:
            return
        self.list_model.set_status(sorted(self.in_flight_rows), "Cancelled")
        self.in_flight_rows.clear()

    def _batch_finished(self, generation: int, _details: tuple) -> None:
        self.worker_running = False
        self.active_worker = None
        if generation != self.generation:
            Qc.QTimer.singleShot(0, self.schedule_next_batch)
            return
        if not self.cancelled:
            Qc.QTimer.singleShot(0, self.schedule_next_batch)

    def closeEvent(self, event: Qg.QCloseEvent) -> None:
        self.abort_requested.emit()
        super().closeEvent(event)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize inputs that maximize Hand TeX classifier outputs."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--symbol", metavar="KEY", help="Show one symbol equivalence class.")
    mode.add_argument("--all", action="store_true", help="Show every model output class.")
    parser.add_argument("--model", type=Path, default=ut.get_model_path())
    parser.add_argument("--encodings", type=Path, default=ut.get_encodings_path())
    parser.add_argument(
        "--database",
        type=Path,
        default=ut.resource_path(training.database, "handtex.db"),
        help="Training stroke database used to derive one loader seed per class.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points-per-stroke", type=int, default=24)
    parser.add_argument("--anchor-weight", type=float, default=20.0)
    parser.add_argument("--shape-weight", type=float, default=0.5)
    parser.add_argument("--exact-interval", type=int, default=10)
    parser.add_argument("--exact-proposals", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = ActivationConfig(
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        noise_standard_deviation=args.noise_standard_deviation,
        batch_size=args.batch_size,
        points_per_stroke=args.points_per_stroke,
        anchor_weight=args.anchor_weight,
        shape_weight=args.shape_weight,
        exact_interval=args.exact_interval,
        exact_proposals=args.exact_proposals,
    )
    try:
        config.validate()
        if not args.database.is_file():
            raise ValueError(f"Stroke database does not exist: {args.database}")
        device = choose_device(args.device)
        symbol_data = sr.SymbolData()
        classifier, encodings = load_visualization_model(args.model, args.encodings, device)

        invalid_encodings = [key for key in encodings if key not in symbol_data]
        non_leaders = [
            key
            for key in encodings
            if key in symbol_data and symbol_data.to_leader.get(key, key) != key
        ]
        if invalid_encodings:
            raise ValueError(
                f"Model encoding is absent from symbol metadata: {invalid_encodings[0]!r}"
            )
        if non_leaders:
            raise ValueError(
                f"Model encoding is not an equivalence-class leader: {non_leaders[0]!r}"
            )

        if args.symbol:
            _, class_index = resolve_symbol_key(args.symbol, symbol_data, encodings)
            class_indices = [class_index]
        else:
            class_indices = list(range(len(encodings)))
        stroke_dataset = build_seed_stroke_dataset(
            args.database,
            symbol_data,
            config.image_size,
            config.seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    app = Qw.QApplication.instance() or Qw.QApplication([parser.prog])
    window = ActivationWindow(
        classifier,
        encodings,
        symbol_data,
        class_indices,
        config,
        device,
        stroke_dataset=stroke_dataset,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
