# DAFQ: Decoupled Accuracy-Efficiency Floating-Point Quantization

## Core Idea

DAFQ proposes a decoupled framework for LLM weight-activation quantization. Instead of treating weight quantization and activation quantization as a single symmetric problem, DAFQ separates them according to their deployment constraints:

- **Weight quantization** is an offline problem, so it should prioritize accuracy.
- **Activation quantization** is an online problem, so it should prioritize efficiency.

The central principle is:

> Optimize weights accuracy-first offline, then optimize activations efficiency-first under the fixed quantized-weight model.

This preserves the most important interaction between weights and activations while avoiding an unnecessarily expensive online activation quantization process.

## Motivation

In LLM inference, weights and activations have different properties:

- Weights are static and can be pre-quantized before deployment.
- Activations are input-dependent and must be quantized during inference.
- Offline weight optimization can afford expensive calibration, search, and reconstruction.
- Online activation quantization must remain lightweight because it happens at every layer and token.

Previous W/A quantization methods often formulate weight and activation quantization jointly. DAFQ argues that this ignores the asymmetric cost structure of deployment.

## Floating-Point Quantization Setting

DAFQ targets low-bit floating-point formats such as:

- MXFP
- NVFP
- HIFP

These formats provide flexible trade-offs through exponent bits, mantissa bits, block size, shared exponent, and scaling strategy. This makes them well suited for a decoupled accuracy-efficiency design.

## Stage 1: Accuracy-First Weight Quantization

For a linear layer:

```math
Y = WX
```

DAFQ first quantizes the weight matrix offline:

```math
\hat{W} = Q_w(W)
```

The weight-stage objective prioritizes reconstruction accuracy:

```math
\min_{Q_w} \mathbb{E}_{X} \|WX - \hat{W}X\|_2^2
```

Because this stage is offline, it can use relatively expensive optimization, including:

- Layer-wise or block-wise floating-point format search.
- Exponent/mantissa allocation.
- Block size or group size selection.
- Shared exponent or scale optimization.
- Clipping threshold search.
- Optional smoothing, rotation, or outlier handling.

### Practical Weight Algorithm

1. Collect calibration activations.
2. For each layer, enumerate candidate floating-point quantization formats.
3. Quantize the weight matrix under each candidate.
4. Measure output reconstruction error using calibration data.
5. Select the candidate with the best accuracy.
6. Optionally refine scale or clipping parameters.

This stage spends computation where it is cheap: before deployment.

## Stage 2: Weight-Aware Activation Importance

After the quantized weights are fixed, activation quantization should be optimized with respect to the quantized-weight model, not the original full-precision model.

The activation-stage reconstruction target is:

```math
\|\hat{W}X - \hat{W}Q_a(X)\|_2^2
```

instead of:

```math
\|X - Q_a(X)\|_2^2
```

This leads to a simple activation-channel importance score:

```math
I_j = \|\hat{W}_{:,j}\|_2^2
```

Channels with larger `I_j` are more sensitive to activation quantization error. Channels with smaller `I_j` can be quantized more aggressively.

The activation quantization objective can be written as:

```math
\min_{Q_a} \sum_j I_j \|x_j - Q_a(x_j)\|_2^2 + \lambda \cdot \mathrm{Cost}(Q_a)
```

This makes activation quantization decoupled from weight optimization, but still conditioned on the fixed quantized weights.

## Stage 3: Efficiency-First Activation Quantization

Activation quantization is performed online, so DAFQ prioritizes runtime efficiency:

```math
\min_{Q_a} \mathrm{OnlineCost}(Q_a)
\quad
\mathrm{s.t.}
\quad
\Delta \mathrm{Accuracy} \le \epsilon
```

The activation quantizer should avoid expensive per-token search. Candidate strategies include the following.

### Static Layer Scaling

Learn one clipping threshold or scale per layer during calibration, then reuse it during inference:

```math
\hat{X} = Q_a(X; s_l)
```

This is the cheapest strategy, but it may be less robust to activation outliers.

### Power-of-Two Dynamic Scaling

Compute a simple dynamic scale from the activation maximum:

```math
s = 2^{\lfloor \log_2(\max |X|) \rfloor}
```

This is hardware-friendly for floating-point formats because scaling can be implemented through exponent adjustment.

### Hybrid Static-Dynamic Scaling

Use a static scale for normal tokens and switch to dynamic scaling only when an outlier is detected:

```math
\text{if } \max |X| > \tau_l:
\quad \text{use dynamic scale}
```

```math
\text{else}:
\quad \text{use static scale}
```

This strategy keeps common-case inference cheap while preserving robustness on difficult tokens.

## Full DAFQ Pipeline

1. Collect calibration data.
2. Quantize weights offline with an accuracy-first objective.
3. Fix the quantized weights.
4. Estimate activation-channel importance from the quantized weights.
5. Calibrate efficient activation quantization policies.
6. During inference, use pre-quantized weights and lightweight online activation quantization.

## Expected Contributions

DAFQ can be positioned around four main contributions:

1. A decoupled formulation of LLM W/A quantization based on offline-online deployment asymmetry.
2. An accuracy-first floating-point weight quantization stage using offline format and scale search.
3. A weight-aware activation quantization stage conditioned on the fixed quantized weights.
4. An efficiency-first online activation policy, such as hybrid static-dynamic scaling.

## Experimental Plan

Recommended baselines:

- Weight-only quantization.
- Joint W/A quantization.
- Same-format floating-point W/A quantization.
- SmoothQuant-style methods.
- AWQ/GPTQ-style weight quantization combined with activation quantization.
- DAFQ with different activation policies.

Recommended metrics:

- Perplexity.
- Downstream task accuracy.
- Inference latency.
- Activation quantization overhead.
- Memory bandwidth.
- Accuracy-efficiency Pareto curves.

Useful ablations:

- Weight format search vs fixed format.
- Activation quantization with and without weight-aware importance.
- Static activation scaling vs dynamic scaling vs hybrid scaling.
- Different MXFP/NVFP/HIFP bit formats.
- Different block sizes or group sizes.

## Key Claim

DAFQ should aim to demonstrate that:

> By spending more computation on offline weight quantization and using lightweight weight-aware activation quantization online, DAFQ achieves a better accuracy-efficiency trade-off than symmetric joint W/A quantization methods.

