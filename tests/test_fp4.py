import torch

from sepquant.formats import (
    HIF4Format,
    HIF4ScaleSearchFormat,
    MXFP4E4M3Format,
    MXFP4E4M3ScaleSearchFormat,
    MXFP4E5M3Format,
    MXFP4E5M3ScaleSearchFormat,
    MXFP4Format,
    MXFP4PlusFormat,
    MXFP4PlusScaleSearchFormat,
    MXFP4ScaleSearchFormat,
    NVFP4Format,
    NVFP4ScaleSearchFormat,
    get_fp4_format,
)
from sepquant.formats.mxfp import _e0m8_macro_scale_code_from_raw


def test_mxfp4_quantize_preserves_shape() -> None:
    fmt = get_fp4_format("mxfp4")
    tensor = torch.randn(3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, MXFP4Format)


def test_mxfp4_e4m3_uses_unsigned_e4m3_block_scale() -> None:
    fmt = get_fp4_format("mxfp4_e4m3")
    tensor = torch.randn(3, 65)

    quantized = fmt.quantize(tensor)

    assert isinstance(fmt, MXFP4E4M3Format)
    assert fmt.block_size == 32
    assert fmt.scale_exponent_bits == 4
    assert fmt.scale_mantissa_bits == 3
    assert fmt.scale_exponent_bias == 7
    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()


def test_mxfp4_e5m3_uses_unsigned_e5m3_block_scale() -> None:
    fmt = get_fp4_format("mxfp4_ue5m3")
    tensor = torch.randn(3, 65)

    quantized = fmt.quantize(tensor)

    assert isinstance(fmt, MXFP4E5M3Format)
    assert fmt.block_size == 32
    assert fmt.scale_exponent_bits == 5
    assert fmt.scale_mantissa_bits == 3
    assert fmt.scale_exponent_bias == 15
    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()


def test_mxfp4_e4m3_and_e5m3_preserve_zeros() -> None:
    for name in ("mxfp4_e4m3", "mxfp4_e5m3"):
        fmt = get_fp4_format(name)
        tensor = torch.zeros(2, 65)

        quantized = fmt.quantize(tensor)

        assert torch.equal(quantized, tensor)


def test_mxfp4_e5m3_represents_smaller_scales_than_e4m3() -> None:
    # Below E4M3 min subnormal (2**-9), E4M3 collapses while E5M3 still has codes.
    tiny = torch.full((1, 32), 2.0**-12)
    e4m3 = get_fp4_format("mxfp4_e4m3")
    e5m3 = get_fp4_format("mxfp4_e5m3")

    e4m3_scale = e4m3._block_scale(tiny)
    e5m3_scale = e5m3._block_scale(tiny)

    assert torch.isclose(e4m3_scale, torch.tensor([[e4m3.eps]], dtype=e4m3_scale.dtype)).all()
    assert e5m3_scale.item() > e4m3.eps


def test_mxfp4_e4m3_search_is_no_worse_than_default_scale() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(4, 65)
    default = get_fp4_format("mxfp4_e4m3").quantize(tensor)
    searched = get_fp4_format("mxfp4_e4m3_search").quantize(tensor)

    default_error = torch.sum((tensor - default).square())
    searched_error = torch.sum((tensor - searched).square())

    assert isinstance(get_fp4_format("mxfp4_e4m3_search"), MXFP4E4M3ScaleSearchFormat)
    assert searched_error <= default_error


def test_mxfp4_e5m3_search_is_no_worse_than_default_scale() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(4, 65)
    default = get_fp4_format("mxfp4_e5m3").quantize(tensor)
    searched = get_fp4_format("mxfp4_e5m3_search").quantize(tensor)

    default_error = torch.sum((tensor - default).square())
    searched_error = torch.sum((tensor - searched).square())

    assert isinstance(get_fp4_format("mxfp4_e5m3_search"), MXFP4E5M3ScaleSearchFormat)
    assert searched_error <= default_error


def test_mxfp4_plus_uses_macro_and_block_scales() -> None:
    fmt = get_fp4_format("mxfp4_plus")
    tensor = torch.randn(2, 3, 129)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, MXFP4PlusFormat)
    assert fmt.block_size == 16
    assert fmt.macro_block_size == 128


def test_mxfp4_plus_alias_and_zeros() -> None:
    fmt = get_fp4_format("mxfp4+")
    tensor = torch.zeros(2, 129)

    quantized = fmt.quantize(tensor)

    assert torch.equal(quantized, tensor)
    assert isinstance(fmt, MXFP4PlusFormat)


def test_mxfp4_plus_mbs_code_uses_reciprocal_scale_truncation() -> None:
    raw_scale = torch.full((1, 8, 1), 1.0 / 1.5)

    code = _e0m8_macro_scale_code_from_raw(raw_scale)

    assert code.item() == 128


def test_mxfp4_plus_search_quantizes_macro_blocks() -> None:
    fmt = get_fp4_format("mxfp4_plus_search")
    tensor = torch.randn(2, 3, 129)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, MXFP4PlusScaleSearchFormat)
    assert fmt.block_size == 16
    assert fmt.macro_block_size == 128


def test_mxfp4_plus_search_is_no_worse_than_default_macro_scale() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(4, 129)
    default = get_fp4_format("mxfp4_plus").quantize(tensor)
    searched = get_fp4_format("mxfp4+_search").quantize(tensor)

    default_error = torch.sum((tensor - default).square())
    searched_error = torch.sum((tensor - searched).square())

    assert searched_error <= default_error


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


def test_hif4_uses_expected_hierarchical_levels() -> None:
    torch.manual_seed(0)
    fmt = get_fp4_format("hif4")
    tensor = torch.randn(2, 3, 128)

    quantized = fmt.quantize(tensor)

    assert isinstance(fmt, HIF4Format)
    assert fmt.block_size == 64
    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()


def test_hif4_quantize_preserves_shape_with_padding() -> None:
    fmt = get_fp4_format("hif4")
    tensor = torch.randn(2, 3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()


def test_hif4_quantize_preserves_zeros() -> None:
    fmt = get_fp4_format("hif4")
    tensor = torch.zeros(2, 65)

    quantized = fmt.quantize(tensor)

    assert torch.equal(quantized, tensor)


def test_hif4_search_quantizes_hidden_dimension_blocks() -> None:
    fmt = get_fp4_format("hif4_search")
    tensor = torch.randn(2, 3, 65)

    quantized = fmt.quantize(tensor)

    assert quantized.shape == tensor.shape
    assert torch.isfinite(quantized).all()
    assert isinstance(fmt, HIF4ScaleSearchFormat)
    assert fmt.block_size == 64


def test_hif4_search_is_no_worse_than_default_level1_scale() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(4, 129)
    default = get_fp4_format("hif4").quantize(tensor)
    searched = get_fp4_format("hif4_search").quantize(tensor)

    default_error = torch.sum((tensor - default).square())
    searched_error = torch.sum((tensor - searched).square())

    assert searched_error <= default_error

