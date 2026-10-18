"""Tests for the activation-maximization analysis applet."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch
from PySide6 import QtWidgets as Qw
from torch import nn

from handtex.structures import Symbol
from handtex.detector.model import CNN
from training.misc_scripts.activation_maximizer import (
    ActivationConfig,
    ActivationWindow,
    build_seed_stroke_dataset,
    build_argument_parser,
    generate_stroke_activation_batch,
    initialize_inputs,
    load_visualization_model,
    resolve_symbol_key,
    resample_polyline,
    render_soft_strokes,
    select_next_rows,
)


class TinyClassifier(nn.Module):
    def __init__(self, output_classes=2):
        super().__init__()
        self.convolution = nn.Conv2d(1, output_classes, kernel_size=1, bias=False)
        with torch.no_grad():
            self.convolution.weight[0].fill_(-1.0)
            if output_classes > 1:
                self.convolution.weight[1].fill_(1.0)

    def forward(self, images):
        return self.convolution(images).mean(dim=(2, 3))


class DummySymbolData:
    def __init__(self):
        self.symbols = {
            "leader": Symbol.dummy("leader"),
            "member": Symbol.dummy("member"),
        }
        self.all_keys = list(self.symbols)
        self.leaders = ["leader"]
        self.to_leader = {"member": "leader"}

    def __contains__(self, key):
        return key in self.symbols

    def __getitem__(self, key):
        return self.symbols[key]


class TinyStrokeDataset:
    def __init__(self, image_size=8, symbol="zero"):
        self.image_size = image_size
        self.symbol = symbol

    def range_for_symbol(self, symbol):
        assert symbol == self.symbol
        return range(0, 1)

    def load_transformed_strokes(self, index):
        assert index == 0
        return [[(0, 0), (1000, 1000)]], self.symbol


def small_config(**changes):
    values = {
        "seed": 11,
        "steps": 8,
        "learning_rate": 0.1,
        "noise_standard_deviation": 0.08,
        "batch_size": 2,
        "image_size": 8,
    }
    values.update(changes)
    return ActivationConfig(**values)


def test_soft_stroke_renderer_is_differentiable():
    sampled = resample_polyline([(1, 1), (6, 6)], 4)
    assert sampled.shape == (4, 2)
    assert np.allclose(sampled[[0, -1]], ((1, 1), (6, 6)))

    points = torch.from_numpy(sampled)[None, None].requires_grad_(True)
    rendered = render_soft_strokes(points, torch.ones(1, 1), image_size=8)
    assert rendered.shape == (1, 1, 8, 8)
    rendered.mean().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_initialization_is_deterministic_per_class_and_seed():
    config = small_config()
    together = initialize_inputs([0, 1], config, "cpu")
    reordered = initialize_inputs([1, 0], config, "cpu")
    assert torch.equal(together[0], reordered[1])
    assert torch.equal(together[1], reordered[0])

    other_seed = initialize_inputs([0], small_config(seed=12), "cpu")
    assert not torch.equal(together[0], other_seed[0])


def test_stroke_generation_uses_training_loader_seed_and_exact_raster():
    model = TinyClassifier()
    config = small_config(steps=2, points_per_stroke=4, exact_interval=1)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    initial_raw_logit = model(initialize_inputs([0], config, "cpu"))[0, 0].item()

    result = generate_stroke_activation_batch(
        model,
        [0],
        ["zero", "one"],
        config,
        stroke_dataset=TinyStrokeDataset(),
    )[0]

    assert result.raw_image.shape == (8, 8)
    assert result.stroke_image.shape == (8, 8)
    assert result.raw_image.dtype == np.uint8
    assert 0 <= result.raw_image.min() <= result.raw_image.max() <= 255
    assert result.raw_logit > initial_raw_logit
    assert result.stroke_count == 1
    assert result.stroke_probability >= result.initial_stroke_probability
    assert result.stroke_logit >= result.initial_stroke_logit
    assert set(np.unique(result.stroke_image)) <= {0, 255}
    for before, after in zip(parameters_before, model.parameters()):
        assert torch.equal(before, after)


def test_seed_dataset_requests_one_nonrandom_loader_sample_per_class(tmp_path, monkeypatch):
    database_path = tmp_path / "strokes.db"
    database_path.touch()
    captured = {}

    class FakeStrokeDataset:
        symbol_keys = ["leader"]

        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "training.misc_scripts.activation_maximizer.dl.StrokeDataset", FakeStrokeDataset
    )

    dataset = build_seed_stroke_dataset(database_path, DummySymbolData(), 64, 11)

    assert isinstance(dataset, FakeStrokeDataset)
    assert captured["kwargs"]["class_limit"] == 1
    assert captured["kwargs"]["random_augmentation"] is False
    assert captured["kwargs"]["debug_single_sample_only"] is True


def test_stroke_activation_generation_is_repeatable():
    config = small_config(steps=2)
    first = generate_stroke_activation_batch(
        TinyClassifier(), [0], ["zero", "one"], config, stroke_dataset=TinyStrokeDataset()
    )[0]
    second = generate_stroke_activation_batch(
        TinyClassifier(), [0], ["zero", "one"], config, stroke_dataset=TinyStrokeDataset()
    )[0]
    assert np.array_equal(first.raw_image, second.raw_image)
    assert np.array_equal(first.stroke_image, second.stroke_image)
    assert first.raw_logit == second.raw_logit


def test_symbol_resolution_maps_members_and_reports_errors():
    symbols = DummySymbolData()
    assert resolve_symbol_key("member", symbols, ["leader"]) == ("leader", 0)
    with pytest.raises(ValueError, match="Unknown symbol key"):
        resolve_symbol_key("missing", symbols, ["leader"])
    with pytest.raises(ValueError, match="absent from the model encodings"):
        resolve_symbol_key("leader", symbols, ["something-else"])


def test_visible_pending_rows_are_prioritized_without_duplicates():
    pending = {0, 1, 2, 4, 7}
    assert select_next_rows(pending, [7, 4, 7, 5], 4) == [7, 4, 0, 1]
    assert select_next_rows({2, 3}, [], 8) == [2, 3]


def test_cli_requires_exactly_one_mode():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--all", "--symbol", "leader"])
    assert parser.parse_args(["--symbol", "leader"]).symbol == "leader"
    assert parser.parse_args(["--all"]).all is True
    configured = parser.parse_args(
        [
            "--all",
            "--anchor-weight",
            "30",
            "--points-per-stroke",
            "12",
        ]
    )
    assert configured.anchor_weight == 30
    assert configured.points_per_stroke == 12


def test_training_checkpoint_dictionary_can_be_loaded(tmp_path):
    encodings_path = tmp_path / "encodings.txt"
    encodings_path.write_text("zero\none\n")
    checkpoint_path = tmp_path / "model.chkpt"
    torch.save({"model_state_dict": CNN(2, 64).state_dict(), "epoch": 1}, checkpoint_path)

    model, encodings = load_visualization_model(
        checkpoint_path, encodings_path, torch.device("cpu")
    )

    assert encodings == ["zero", "one"]
    assert model.fc1.out_features == 2
    assert model.training is False


def test_offscreen_window_completes_and_can_regenerate():
    app = Qw.QApplication.instance() or Qw.QApplication([])
    window = ActivationWindow(
        TinyClassifier(output_classes=1),
        ["leader"],
        DummySymbolData(),
        [0],
        small_config(steps=1, image_size=64, batch_size=1, exact_proposals=0),
        torch.device("cpu"),
        TinyStrokeDataset(image_size=64, symbol="leader"),
    )

    deadline = time.monotonic() + 15
    while not window.completed_rows and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.completed_rows == {0}
    first_generation = window.generation

    window.regenerate()
    assert window.generation == first_generation + 1
    assert window.list_model.entries[0].result is None
    window.cancel_generation()
    window.thread_pool.waitForDone(5000)
    app.processEvents()
    window.close()
