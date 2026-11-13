#!/usr/bin/env python3
"""Regenerate src/lut.rs from tinymind's cpp/lookupTables.cpp.

Tables are copied by script, never typed by hand. For each table this script
asserts the element count, the C++ element type and that every value fits it,
then writes src/lut.rs and runs rustfmt on it. Stdlib only; run it with
`python3 -I scripts/extract-luts.py` from the repository root. TINYMIND_DIR
overrides the tinymind checkout (default ../tinymind).

To port another table, add a row to TABLES.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINYMIND_DIR = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
SOURCE = os.path.join(TINYMIND_DIR, "cpp", "lookupTables.cpp")
OUTPUT = os.path.join(ROOT, "src", "lut.rs")
COUNT = 96  # NUMBER_OF_ACTIVATION_TABLE_VALUES in cpp/activation.hpp

# (Rust name, C++ guard macro, C++ table struct, C++ element type, Rust element type)
TABLES = [
    ("TANH_Q16_16", "TINYMIND_USE_TANH_16_16", "TanhValuesTableQ16_16", "uint32_t", "u32"),
    ("SIGMOID_Q16_16", "TINYMIND_USE_SIGMOID_16_16", "SigmoidValuesTableQ16_16", "uint32_t", "u32"),
    ("TANH_Q8_8", "TINYMIND_USE_TANH_8_8", "TanhValuesTableQ8_8", "uint16_t", "u16"),
    ("SIGMOID_Q8_8", "TINYMIND_USE_SIGMOID_8_8", "SigmoidValuesTableQ8_8", "uint16_t", "u16"),
    ("EXP_Q16_16", "TINYMIND_USE_EXP_16_16", "ExpValuesTableQ16_16", "uint32_t", "u32"),
    ("EXP_Q8_8", "TINYMIND_USE_EXP_8_8", "ExpValuesTableQ8_8", "uint16_t", "u16"),
]
BITS = {"u16": 16, "u32": 32}


def extract(text, guard, struct, ctype):
    block = re.search(
        r"#if %s\b(.*?)#endif // %s\b" % (guard, guard), text, re.S
    )
    if not block:
        sys.exit("guard %s not found in %s" % (guard, SOURCE))
    body = block.group(1)
    decl = "const %s %s::values[NUMBER_OF_ACTIVATION_TABLE_VALUES] = {" % (ctype, struct)
    if decl not in body:
        sys.exit("declaration %r not found under %s" % (decl, guard))
    inner = body.split(decl, 1)[1].split("};", 1)[0]
    values = re.findall(r"0x[0-9A-Fa-f]+", inner)
    leftover = re.sub(r"0x[0-9A-Fa-f]+|[\s,]", "", inner)
    if leftover:
        sys.exit("unexpected text in %s: %r" % (struct, leftover[:80]))
    return values


def main():
    with open(SOURCE) as f:
        text = f.read()

    out = [
        "//! Activation lookup tables generated from tinymind (C++) `cpp/lookupTables.cpp` by",
        "//! `scripts/extract-luts.py`. Do not edit by hand; rerun the script.",
        "",
        "pub const NUMBER_OF_ACTIVATION_TABLE_VALUES: usize = %d;" % COUNT,
    ]
    for name, guard, struct, ctype, rtype in TABLES:
        values = extract(text, guard, struct, ctype)
        if len(values) != COUNT:
            sys.exit("%s: %d values, expected %d" % (struct, len(values), COUNT))
        for v in values:
            if int(v, 16) >= 1 << BITS[rtype]:
                sys.exit("%s: %s does not fit %s" % (struct, v, rtype))
        print("%-15s %d values, first %s, last %s" % (name, len(values), values[0], values[-1]))
        out += [
            "",
            "pub static %s: [%s; NUMBER_OF_ACTIVATION_TABLE_VALUES] = [" % (name, rtype),
            *("    %s," % v for v in values),
            "];",
        ]

    with open(OUTPUT, "w") as f:
        f.write("\n".join(out) + "\n")
    subprocess.run(["rustfmt", "--edition", "2024", OUTPUT], check=True)
    print("wrote", os.path.relpath(OUTPUT, ROOT))


if __name__ == "__main__":
    main()
