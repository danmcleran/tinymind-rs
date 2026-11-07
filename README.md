# rustymind

rustymind brings [tinymind](https://github.com/danmcleran/tinymind), a
header-only C++ library for fixed-point neural networks on embedded systems, to
Rust. Like tinymind, it is free, open source software under the MIT license. The
library is `no_std` and allocation-free, and every ported path is verified
bit-for-bit against the C++ build: the XOR example's weights, training log and
decision surface are byte-identical to the C++ output.

## Status

Early. So far the port covers what tinymind's XOR example uses: Q16.16
arithmetic, lookup-table tanh and sigmoid, and a trainable one-hidden-layer
perceptron. The API is not stable yet, and the crate is not on crates.io yet.

## Quick start

```bash
cargo run --release --bin xor -- output   # writes the output files into ./output
cargo test                                # golden test against the C++ output
scripts/compare-with-cpp.sh               # rebuild the C++ example and diff everything
```

The `xor` binary takes an optional output directory (default `output`), then an
optional learning rate and momentum. It writes `nn_fixed_xor.txt`,
`xor_training.csv` and `xor_decision_surface.csv` there and prints a summary on
stdout.

`scripts/compare-with-cpp.sh` needs a tinymind checkout and `g++`. It looks in
`../tinymind` unless `TINYMIND_DIR` is set.

## What is ported

| Rust | C++ (tinymind) |
| --- | --- |
| `src/qformat.rs` | `cpp/qformat.hpp`: `QValue`, Truncate and Wrap policies, `i32` storage only |
| `src/activation.rs` | `cpp/activationFunctions.hpp`, `cpp/lookupTable.hpp`, `cpp/interpolate.hpp` |
| `src/lut.rs` | `cpp/lookupTables.cpp`: Q16.16 tanh and sigmoid tables |
| `src/neuralnet.rs` | `cpp/neuralnet.hpp`: one-hidden-layer MLP, SGD with momentum, batch size 1 |
| `src/rand.rs` | a re-implementation of glibc `rand()`, which the C++ example uses for weight init and sampling |
| `src/bin/xor.rs` | `examples/xor/xor.cpp` |

Everything else in tinymind (other value types, recurrent layers, the int8
quantization tiers, SIMD backends, and so on) is not ported.

## How equivalence is proven

- `tests/xor_golden.rs` runs the `xor` binary and compares its stdout and the
  two CSV files byte-for-byte with the goldens in `tests/golden/`. The 2.6 MB
  `nn_fixed_xor.txt` is checked by length, line count and a 64-bit FNV-1a hash
  instead of being stored. The goldens come from the C++ build; `TINYMIND_REF` records the tinymind commit they were last verified against.
- `scripts/compare-with-cpp.sh` builds the C++ example from a tinymind checkout
  with the flags the C++ build uses, runs both programs, and runs `cmp` on all
  four output files. It exits nonzero if any file differs.

## License

MIT. See `LICENSE` and `NOTICE`.
