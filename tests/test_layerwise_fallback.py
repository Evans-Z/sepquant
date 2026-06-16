import json

from torch import nn

from sepquant.models import TargetLinear
from sepquant.optimization.layerwise import optimize_layers


class FailingOptimizer:
    name = "failing"

    def optimize(self, _context):
        raise AssertionError("under-sampled fallback should avoid optimizer call")


def test_optimize_layers_uses_fallback_for_under_sampled_layer(tmp_path) -> None:
    calibration_dir = tmp_path / "calib"
    calibration_dir.mkdir()
    (calibration_dir / "metadata.json").write_text(
        json.dumps(
            {
                "token_counts": {
                    "layer": 0,
                },
                "layers": {},
            }
        ),
        encoding="utf-8",
    )

    results = optimize_layers(
        targets=[TargetLinear(name="layer", module=nn.Linear(4, 4))],
        calibration_dir=calibration_dir,
        optimizer=FailingOptimizer(),
        min_tokens_per_layer=128,
        fallback_weight_format="mxfp4_search",
        fallback_activation_format="none",
        fallback_rotation="none",
    )

    assert len(results) == 1
    result = results[0]
    assert result.layer_name == "layer"
    assert result.spec.enabled
    assert result.spec.weight_format == "mxfp4_search"
    assert result.metrics["reason"] == "insufficient_calibration_tokens"
    assert result.metrics["token_count"] == 0


def test_optimize_layers_can_disable_under_sampled_layer(tmp_path) -> None:
    calibration_dir = tmp_path / "calib"
    calibration_dir.mkdir()
    (calibration_dir / "metadata.json").write_text(
        json.dumps(
            {
                "token_counts": {
                    "layer": 1,
                },
                "layers": {},
            }
        ),
        encoding="utf-8",
    )

    results = optimize_layers(
        targets=[TargetLinear(name="layer", module=nn.Linear(4, 4))],
        calibration_dir=calibration_dir,
        optimizer=FailingOptimizer(),
        min_tokens_per_layer=128,
        fallback_weight_format="none",
    )

    assert len(results) == 1
    assert not results[0].spec.enabled
    assert results[0].spec.weight_format is None
