import torch

from sepquant.formats import (
    MXFP4Format,
    MXFP4ScaleSearchFormat,
    NVFP4Format,
    NVFP4ScaleSearchFormat,
    get_fp4_format,
)


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


def test_nvfp4_search_quantizes_hidden_dimension_blocks() -> None:
    fmt = get_fp4_format("nvfp4_search")
    tensor = torch.randn(2, 3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, NVFP4ScaleSearchFormat)
    assert fmt.block_size == 16


def test_nvfp4_search_is_no_worse_than_default_scale() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(4, 65)
    default = get_fp4_format("nvfp4").quantize(tensor)
    searched = get_fp4_format("nvfp4_search").quantize(tensor)

    default_error = torch.sum((tensor - default).square())
    searched_error = torch.sum((tensor - searched).square())

    assert searched_error <= default_error

