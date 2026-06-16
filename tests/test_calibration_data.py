import torch

from sepquant.calibration import data


class TinyTokenizer:
    def __call__(self, _text: str, *, return_tensors: str):
        assert return_tensors == "pt"
        return type("Encoded", (), {"input_ids": torch.arange(40).unsqueeze(0)})()


def test_build_calibration_batches_groups_samples(monkeypatch) -> None:
    monkeypatch.setattr(data, "load_dataset", lambda *args, split: {"text": ["ignored"]})

    batches = data.build_calibration_batches(
        tokenizer=TinyTokenizer(),
        dataset_name="dummy",
        dataset_config=None,
        split="train",
        text_column="text",
        nsamples=5,
        seed=0,
        sequence_length=4,
        batch_size=2,
    )

    assert [tuple(batch.shape) for batch in batches] == [(2, 4), (2, 4), (1, 4)]


def test_build_calibration_batches_rejects_invalid_batch_size(monkeypatch) -> None:
    monkeypatch.setattr(data, "load_dataset", lambda *args, split: {"text": ["ignored"]})

    try:
        data.build_calibration_batches(
            tokenizer=TinyTokenizer(),
            dataset_name="dummy",
            dataset_config=None,
            split="train",
            text_column="text",
            nsamples=1,
            seed=0,
            sequence_length=4,
            batch_size=0,
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Expected batch_size validation error")
