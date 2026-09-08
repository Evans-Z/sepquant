import argparse

from sepquant.evaluation.multimodal import build_lmms_eval_command


def test_lmms_eval_command_targets_qwen3_vl_checkpoint() -> None:
    args = argparse.Namespace(
        model="outputs/checkpoints/qwen3-vl",
        tasks=["mme", "mmmu_val"],
        batch_size="1",
        limit=8,
        device="cuda:0",
        output_path=None,
        model_args={"max_pixels": 1024},
        log_samples=False,
    )

    command = build_lmms_eval_command(args)

    assert command[command.index("--model") + 1] == "qwen3_vl"
    assert "pretrained=outputs/checkpoints/qwen3-vl,max_pixels=1024" in command
    assert command[command.index("--tasks") + 1] == "mme,mmmu_val"
    assert command[command.index("--limit") + 1] == "8"
