# Synthesizable Fixed-Point Scaled Dot-Product Attention

This repository implements and verifies a compact scaled dot-product attention accelerator in synthesizable SystemVerilog:

```text
Q = X WQ,  K = X WK,  V = X WV
S = Q K^T / sqrt(d_k)
A = approximate_softmax(S)
O = A V
```

The default demonstrator uses `N=2` tokens, embedding/key/value dimension `D=4`, signed INT8 input/weight files, a single time-multiplexed MAC datapath, and a hardware-friendly softmax approximation. The RTL contains no real-number arithmetic, delays, file I/O, or other simulation-only constructs. File loading, result checking, text output, and VCD dumping are isolated in the testbench.

## Repository layout

```text
rtl/attention_core.sv   Synthesizable accelerator and controller
tb/tb_attention.sv      Self-checking SystemVerilog testbench
scripts/golden_model.py Bit-accurate model, float comparison, vector generator
data/*.hex              Flattened row-major input and expected-output files
results/output.txt      Generated signed fixed-point output matrix
results/attention.vcd   Generated waveform
Makefile                Reproducible simulation and verification flow
```

## Quick start

Requirements: Python 3, Icarus Verilog, and `make`.

```sh
make test
```

The command regenerates deterministic vectors, compiles the RTL with SystemVerilog-2012 enabled, runs the self-checking testbench, checks the output again with Python, and verifies that non-empty output and VCD files exist. A successful run ends with both messages:

```text
PASS: all 8 output elements match bit-exact golden data
PASS: results/output.txt matches the bit-accurate Python model
```

Useful individual commands:

```sh
make vectors   # regenerate data/*.hex
make sim       # compile and run the HDL testbench
make verify    # simulate, run Python check, and check generated artifacts
make clean     # remove generated build/ and results/
python3 scripts/golden_model.py   # print all reference intermediates
```

Open `results/attention.vcd` in GTKWave if desired:

```sh
gtkwave results/attention.vcd
```

## Small numerical example

All values below are shown as stored integers. Divide Q/K/V/scores/outputs by 16 and attention weights by 256 to obtain their real values.

```text
X = [[16,  0, 16,  0],     WQ = WK = 16 * identity
     [ 0, 16,  0, 16]]

WV = [[16,  8,  0,  0],
      [ 0, 16,  8,  0],
      [ 0,  0, 16,  8],
      [ 8,  0,  0, 16]]

Q = K = [[16,  0, 16,  0],
         [ 0, 16,  0, 16]]

V = [[16,  8, 16,  8],
     [ 8, 16,  8, 16]]

S = [[16,  0],             A(Q0.8) = [[187,  69],
     [ 0, 16]]                         [ 69, 187]]

O = [[14, 10, 14, 10],
     [10, 14, 10, 14]]
```

For this example, the largest absolute output error versus ideal floating-point attention is `0.009471` in real-value units.

## Architecture

```text
 load interface
      |
  X/WQ/WK/WV memories
      |
      v
 time-multiplexed signed MAC <---- controller FSM
      |              |                 |
   Q/K/V buffers  score buffer     loop indices
                         |
              row max -> exp LUT -> integer normalization
                         |
                   weight buffer
                         |
                  output MAC -> O buffer -> read port
```

The FSM executes nine phases: `IDLE`, `Q`, `K`, `V`, `SCORE`, `ROW_MAX`, `EXP`, `NORMALIZE`, `OUTPUT`, and `DONE`. One dot-product term is consumed per cycle. This intentionally favors low area and clear control over throughput. The default compute takes 140 state-work cycles plus one cycle to raise `done` after `start` is accepted; input streaming is separate.

Approximate cycle count for square `D` matrices is:

```text
3*N*D*D + N*N*D + 3*N*N + N*N*D + 1(done)
```

The first term forms Q/K/V, the second forms scores, `3*N*N` performs max/LUT/normalization, and the fourth forms O.

## Fixed-point formats and bit growth

| Quantity | Stored format | Meaning |
|---|---:|---|
| X, WQ, WK, WV | signed 8-bit Q4.4 | integer value / 16 |
| Q, K, V, O | signed 16-bit Q12.4 | integer value / 16 |
| score S | signed 24-bit, 4 fractional bits | integer value / 16 |
| `1/sqrt(D)` constant | unsigned Q0.8 | default `128/256 = 0.5` |
| exp LUT result | unsigned Q0.8 | 0 through 256 |
| normalized weight | unsigned Q0.8 | 0 through 256 |
| accumulators | signed 64-bit | conservative guard width |

Matrix multiplication produces a Q8.8 product. The accumulator retains all product fractional bits, then `round_shift(..., 4)` returns to four fractional bits. Scores multiply two Q12.4 values (eight fractional bits), then multiply by the Q0.8 scale constant and shift by 12 to return to four fractional bits. Output accumulation multiplies Q0.8 weights by Q12.4 values and shifts by eight.

Rounding is symmetric round-to-nearest with ties away from zero. Q/K/V, score, and output writes saturate to their destination widths instead of wrapping. The 64-bit accumulator makes overflow impossible for the supplied dimensions; for much larger parameters, its bound must be re-evaluated. A conservative Q/K/V dot-product bound is `D * (2^(VEC_W-1))^2`.

## Softmax approximation

Each score row first subtracts its maximum. This is the standard numerical-stability transform and guarantees at least one LUT input is zero. A 12-entry piecewise-constant LUT approximates `exp(x)` from 0 to -5.5 in steps of 0.5, returning Q0.8 values:

```text
x:       0  -0.5  -1.0  -1.5  -2.0  ...  -5.5  below -5.75
exp_Q8: 256   155    94     57     35  ...     1       0
```

The weights use rounded integer normalization:

```text
weight_i = round(exp_i * 256 / sum(exp))
```

This compact design uses a synthesizable integer `/` operator. It is excellent for a concise verified prototype, but a production accelerator would normally replace the inferred combinational divider with a reciprocal LUT plus multiply, Newton iteration, or a multi-cycle divider. Independent rounding means a row can differ slightly from an exact sum of 256; the output model intentionally reproduces that behavior.

## Input and output files

Each `.hex` file contains one two's-complement word per line in flattened row-major order. X has `N*D` bytes; each weight matrix has `D*D` bytes. For example, Q4.4 values `1.0`, `0.5`, and `-1.0` are encoded as `10`, `08`, and `f0`.

The testbench reads all input files with `$readmemh`, streams them into the synthesizable load interface, starts the core, waits with a timeout, and reads the output port. It writes `results/output.txt` as a signed integer matrix and dumps all signals to `results/attention.vcd`. It also compares every output element against `data/expected_o.hex` and exits nonzero on a mismatch.

## Parameter changes

`N`, `D`, widths, and fractional widths are RTL parameters. When changing `D`, also set:

```text
SCALE_MUL_Q = round(256 / sqrt(D))
```

The current testbench and Python model intentionally fix `N=2, D=4` to provide one reproducible interview example. To create another configuration, update their constants and matrices together. The core loop structure itself does not assume the default dimensions. Ensure `$clog2(N*D)` is nonzero (practically, `N*D >= 2`) and re-check accumulator bounds for large designs.

## Design trade-offs and extension path

- **Area vs latency:** one reused MAC minimizes multiplier count but takes one cycle per dot-product term. Multiple lanes or a systolic array can raise throughput.
- **Precision vs storage/bandwidth:** INT8 Q4.4 inputs align with quantized inference and reduce memory traffic. Wider Q/K/V buffers protect accumulated precision.
- **Softmax accuracy vs cost:** row-max subtraction plus a small LUT is stable and cheap. Finer LUT spacing, interpolation, or base-2 exponentiation improves accuracy at added area.
- **Divider cost:** the prototype's `/` keeps behavior transparent. A reciprocal unit is the most important production optimization.
- **Buffering:** simple register arrays make the example portable. Larger N/D should use inferred SRAM/BRAM and tiled dataflow.
- **Pipelining:** this FSM is non-pipelined. Registering multiplier outputs and overlapping load/compute would improve frequency and utilization.

## Interview connection to INT8 accelerator research

This implementation directly connects prior INT8 fixed-point and efficient-accelerator experience to attention hardware: explicit quantization scales are carried across every operator; accumulator growth, requantization, rounding, saturation, and overflow bounds are first-class architectural choices; row-max subtraction stabilizes a low-precision softmax; and the sequential MAC exposes an easy baseline for discussing data reuse, tiling, PE arrays, memory bandwidth, and throughput/area scaling. A useful interview narrative is to present this verified scalar baseline first, quantify its error and latency, and then explain how the same bit-accurate contract can support a pipelined or systolic implementation without changing the reference model.

PEANO-ViT is a particularly relevant comparison point because it targets power-efficient approximations of Vision Transformer nonlinearities, including softmax. Its approximation ideas can replace the baseline exp LUT/normalizer here while retaining the surrounding Q/K/V, score, output, and verification infrastructure. In an interview, describe PEANO-ViT as prior-art motivation or as a separately measured extension; do not claim that this repository reproduces its complete datapath or published power results.

## Verification strategy

Verification has three independent layers:

1. `golden_model.py` regenerates inputs and expected results with the exact RTL rounding, saturation, LUT, and normalization rules.
2. The HDL testbench compares every DUT output against `expected_o.hex` and has a bounded timeout.
3. Python parses the generated `output.txt`, checks exact matrix equality, reports intermediate tensors, and compares the quantized output against ideal floating point.

The test is deterministic and uses only the Python standard library.

## Synthesis notes

`rtl/attention_core.sv` is synthesizable SystemVerilog. The testbench, file I/O, and VCD calls are intentionally separate. Exact inferred resources depend on synthesis settings: array memories may become registers for this tiny instance, mutually exclusive multiply expressions may share a DSP after optimization, and integer normalization may infer a large divider. Synthesis reports—not RTL operator counts—should be used for final area and timing claims.

## License

MIT; see `LICENSE`.

## Reference

M. E. Sadeghi, A. Fayyazi, S. Azizi, and M. Pedram, “PEANO-ViT: Power-Efficient Approximations of Non-Linearities in Vision Transformers,” *ISLPED*, 2024. [arXiv:2406.14854](https://arxiv.org/abs/2406.14854), [DOI: 10.1145/3665314.3670843](https://doi.org/10.1145/3665314.3670843).
