---
title: Q-format in Rust
layout: default
nav_order: 7
---

# Q-format in Rust

A Q-format value is an integer read as a fixed-point number: Q16.16 stores 16 integer bits (including the sign) and 16 fractional bits in an `i32`, so the raw value 65536 means 1.0. For the background and the full C++ feature set, see tinymind's [Q-Format page](https://danmcleran.github.io/tinymind/q-format.html). This page covers what the Rust port implements and where it differs.

## The type

```rust
pub struct QValue<S, const FRAC: u32>(S);

pub type Q16_16 = QValue<i32, 16>;
pub type Q8_8 = QValue<i16, 8>;
```

`S` is the storage type and `FRAC` the number of fractional bits.

| Storage `S` | Wide type for `*` and `/` | Aliases with lookup tables |
|---|---|---|
| `i32` | `i64` | `Q16_16` |
| `i16` | `i32` | `Q8_8` |

The C++ `QValue<FixedBits, FractionalBits>` chooses its storage type from the bit counts at compile time. Stable Rust cannot compute a type from const-generic arithmetic, so in Rust the storage type is an explicit parameter. A macro generates the methods and operators for each storage type, the same way the Rust standard library implements its integer methods.

Any `FRAC` from 1 to one less than the storage width compiles (an invalid one is a compile-time error), but tanh and sigmoid exist only for the formats tinymind has lookup tables for in this port: `Q16_16` and `Q8_8`. Using them on another format is a compile error, as it is in C++ when the matching table is not compiled in.

## Arithmetic

The port implements tinymind's default policies, `TruncatePolicy` and `WrapPolicy`, with exactly the C++ results:

| Operation | Behaviour |
|---|---|
| `a + b`, `a - b` | Wrap modulo the storage width |
| `a * b` | Product in the wide type, arithmetic shift right by `FRAC` (rounds toward negative infinity), wrap to the storage type |
| `a / b` | `(a × 2^FRAC) / b` in the wide type, truncating toward zero |
| `a / 0` | The storage type's maximum for a positive `a`, its minimum for a negative `a`, 0 for 0 |
| `a.div_int(n)` | Division by a whole number, computed in `i64`; not the same as `a / QValue::from_int(n)` |
| `QValue::from_int(n)` | `n` truncated to the integer bits: in Q8.8, 200 becomes -56 |
| comparisons | On the raw integer |

`Display` prints the raw integer, as the C++ `operator<<` does.

## Activation lookup tables

Tanh and sigmoid read 96-entry tables covering [-6, 6] in steps of 1/8, with linear interpolation between entries and clamping outside the range. The tables are generated from tinymind's `cpp/lookupTables.cpp` by `scripts/extract-luts.py`, never typed by hand, and CI checks that they still match tinymind. For how the tables are built, see tinymind's [Activation LUTs page](https://danmcleran.github.io/tinymind/activation-luts.html).

The [API reference]({{ site.baseurl }}/api/tinymind/qformat/) documents every method.
