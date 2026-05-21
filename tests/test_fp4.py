import torch

from sepquant.formats import MXFP4Format, MXFP4ScaleSearchFormat, NVFP4Format, get_fp4_format


def test_mxfp4_quantize_preserves_shape() -> None:
    fmt = get_fp4_format("mxfp4")
    tensor = torch.randn(3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, MXFP4Format)


def test_nvfp4_quantize_preserves_zeros() -> None:
    fmt = get_fp4_format("nvfp4")
    tensor = torch.zeros(2, 17)

    quantized = fmt.quantize(tensor)

    assert torch.equal(quantized, tensor)


def test_nvfp4_uses_two_level_scale_mode() -> None:
    fmt = get_fp4_format("nvfp4")

    assert isinstance(fmt, NVFP4Format)
    assert fmt.block_size == 16


def test_mxfp4_search_quantizes_hidden_dimension_blocks() -> None:
    fmt = get_fp4_format("mxfp4_search")
    tensor = torch.randn(2, 3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, MXFP4ScaleSearchFormat)

