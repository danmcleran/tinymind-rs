#!/usr/bin/env python3
"""The ONNX importer's worked examples: the models in tools/import_onnx/models
(see make_models.py), quantized by ONNX Runtime.

  python3 -I tools/import_onnx/example_onnx.py --model cnn \\
      --rust examples/common/imported_onnx_cnn_model.rs
  python3 -I tools/import_onnx/example_onnx.py --model cnn --cpp DIR
  python3 -I tools/import_onnx/example_onnx.py --model cnn_int4 \\
      --rust examples/common/imported_onnx_cnn_int4_model.rs

The variants (cnn_int4, resnet_int4, cnn_16x8, resnet_qop_16x8) run a model
with tinymind (C++)'s code generator's options: int4 weights for every dense
and convolution layer (int4_dense, int4_conv), or int16 activations with the
int8 weights (activation_bits=16). For them, --cpp writes
DIR/onnx_<variant>.hpp with tinymind (C++)'s own generator
(apps/import_onnx/tinymind_model_codegen.py), whole: weights, layer objects,
memory plan and forward().

--rust writes the Rust module with this repository's importer. --cpp writes
what tinymind (C++) runs for the same model: DIR/onnx_<model>_weights.hpp from
tinymind (C++)'s own importer (apps/import_onnx/tinymind_import_onnx.py in the
checkout at $TINYMIND_DIR, default ../tinymind), and DIR/onnx_<model>_net.hpp,
a forward pass over tinymind (C++)'s QConv2D / QConv2DPerChannel /
QDepthwiseConv2D / QMaxPool2D / QAdd / QDense layers, its LUT builders,
buildQAddParams and QSoftmaxFixedPoint1D, reading every constant from that
header and wiring each layer to the ones its record's `sources` name.
scripts/cpp/imported_onnx.cpp runs the five networks and
prints what examples/imported_onnx.rs prints; scripts/compare-with-cpp.sh
byte-compares the two. Needs numpy and onnx.
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tinymind_import_onnx as ti  # noqa: E402

MODELS = {
    "cnn": "Conv 3x2 (stride 1x2) + ReLU -> pointwise Conv + ReLU -> Conv 3x3 (stride 2) + ReLU\n"
           "-> Linear + Tanh -> Linear + Sigmoid -> Linear -> Softmax on a 2 x 8 x 10 input\n"
           "(tools/import_onnx/models/cnn.onnx: QDQ format, per-channel weights).",
    "cnn_qop": "Conv 3x3 + ReLU -> pointwise Conv + ReLU -> Linear + ReLU -> Linear without bias\n"
               "-> Sigmoid on a 1 x 6 x 6 input\n"
               "(tools/import_onnx/models/cnn_qop.onnx: QOperator format, per-tensor weights).",
    "mlp": "MatMul + bias Add -> Tanh -> Linear -> Softmax on 7 inputs\n"
           "(tools/import_onnx/models/mlp.onnx: QDQ format, per-tensor weights).",
    "resnet": "Conv 3x3 + ReLU -> MaxPool 2x2 -> a residual block (depthwise Conv 3x3 + ReLU\n"
              "-> pointwise Conv, added to the Tanh of the block's input) -> Sigmoid -> AvgPool 2x2\n"
              "-> global average pool -> Linear on a 2 x 8 x 24 input\n"
              "(tools/import_onnx/models/resnet.onnx: QDQ format, per-channel weights).",
    "resnet_qop": "The same network ending in a Softmax\n"
                  "(tools/import_onnx/models/resnet_qop.onnx: QOperator format, per-tensor weights).",
}


# variant -> (model, options of emit_rust_module / tinymind (C++)'s
# emit_model_header, the C++ generator's nested namespace)
VARIANTS = {
    "cnn_int4": ("cnn", dict(int4_dense=True, int4_conv=True), "int4"),
    "resnet_int4": ("resnet", dict(int4_dense=True, int4_conv=True), "int4"),
    "cnn_16x8": ("cnn", dict(activation_bits=16), "int16x8"),
    "resnet_qop_16x8": ("resnet_qop", dict(activation_bits=16), "int16x8"),
}


def model_path(model: str) -> str:
    return os.path.join(HERE, "models", f"{model}.onnx")


def write_cpp_variant(variant: str, out_dir: str) -> None:
    """DIR/onnx_<variant>.hpp (and its weights header) from tinymind (C++)'s own
    code generator, run in a fresh interpreter so its importer module, which
    shares this one's name, is the checkout's."""
    import subprocess
    model, opts, nested = VARIANTS[variant]
    tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
    script = (
        "import sys\n"
        f"sys.path.insert(0, {os.path.join(tm, 'apps', 'import_onnx')!r})\n"
        "import tinymind_model_codegen as cg\n"
        f"cg.import_onnx_model_with_code({model_path(model)!r}, "
        f"{os.path.join(out_dir, f'onnx_{variant}_weights.hpp')!r}, "
        f"{os.path.join(out_dir, f'onnx_{variant}.hpp')!r}, {('onnx_' + variant)!r}, "
        f"nested_namespace={nested!r}, **{opts!r})\n")
    subprocess.run([sys.executable, "-I", "-B", "-c", script], check=True)


def cpp_importer():
    """tinymind (C++)'s importer module, from its checkout."""
    tm = os.environ.get("TINYMIND_DIR", os.path.join(ROOT, "..", "tinymind"))
    path = os.path.join(tm, "apps", "import_onnx", "tinymind_import_onnx.py")
    spec = importlib.util.spec_from_file_location("tinymind_cpp_import_onnx", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def _out_size(layer, sizes) -> int:
    g = layer.geometry
    if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
        return g["OutputHeight"] * g["OutputWidth"] * g["NumFilters"]
    if layer.op_type == "MaxPool":
        return g["OutputHeight"] * g["OutputWidth"] * g["InputChannels"]
    if layer.op_type == "Dense":
        return g["NumOutputs"]
    return sizes[layer.sources[0]]


def write_cpp(model: str, out_dir: str) -> None:
    """Every layer reads the buffers its `sources` name (the model's input
    for -1), as tinymind (C++)'s importer tests wire a model."""
    cpp = cpp_importer()
    ns = f"onnx_{model}"
    header = f"{ns}_weights.hpp"
    layers = cpp.import_onnx_model(model_path(model), os.path.join(out_dir, header), ns)
    first = layers[0]
    g = first.geometry
    in_size = g["NumInputs"] if first.op_type == "Dense" else \
        g["InputHeight"] * g["InputWidth"] * g["InputChannels"]
    sizes = {-1: in_size}
    lines = [f"// Generated by tools/import_onnx/example_onnx.py --cpp: the {model} example",
             f"// on tinymind (C++)'s layers, every constant from {header} (tinymind (C++)'s",
             "// ONNX importer). Do not edit.",
             "#pragma once",
             "",
             f'#include "{header}"',
             "",
             f"namespace {ns}_net",
             "{",
             f"    namespace w = {ns};",
             f"    constexpr std::size_t InputSize = {in_size};",
             "",
             "    typedef tinymind::Requantizer<int32_t, int8_t> Rq;",
             "",
             "    inline Rq makeRq(int32_t multiplier, int32_t shift, int32_t zeroPoint)",
             "    {",
             "        Rq r;",
             "        r.multiplier = multiplier;",
             "        r.shift = shift;",
             "        r.zero_point = static_cast<int8_t>(zeroPoint);",
             "        r.qmin = -128;",
             "        r.qmax = 127;",
             "        return r;",
             "    }",
             "",
             "    inline void forward(const int8_t* input, int8_t* output)",
             "    {"]

    def buf(j):
        return "input" if j < 0 else f"buf{j}"

    for i, layer in enumerate(layers):
        k = f"w::k{cpp.sanitize(layer.name)}"
        g = layer.geometry
        size = _out_size(layer, sizes)
        sizes[i] = size
        x = buf(layer.sources[0])
        lines.append(f"        static int8_t buf{i}[{size}];")
        if layer.weight is not None:
            n_out = layer.weight.shape[0]
            if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
                dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                        f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}")
                pads = f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}"
            if layer.op_type == "Conv" and layer.per_channel:
                ty = f"QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, {dims}, {g['NumFilters']}, {pads}>"
            elif layer.op_type == "Conv":
                ty = f"QConv2D<int8_t, int8_t, int32_t, int8_t, {dims}, {g['NumFilters']}, {pads}>"
            elif layer.op_type == "Dense" and layer.per_channel:
                ty = (f"QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, "
                      f"1, 1, {g['NumInputs']}, 1, 1, 1, 1, {g['NumOutputs']}>")
            elif layer.op_type == "Dense":
                ty = f"QDense<int8_t, int8_t, int32_t, int8_t, {g['NumInputs']}, {g['NumOutputs']}>"
            else:  # DepthwiseConv, AveragePool: one requantizer per channel
                ty = f"QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, {dims}, {pads}>"
            if layer.per_channel or ty.startswith("QDepthwiseConv2D"):
                rq = (f"{k}_Multipliers[c], {k}_Shifts[c]" if layer.per_channel
                      else f"{k}_Multiplier, {k}_Shift")
                lines += [f"        static Rq rq{i}[{n_out}];",
                          f"        for (std::size_t c = 0; c < {n_out}; ++c)",
                          "        {",
                          f"            rq{i}[c] = makeRq({rq}, {k}_OutputZeroPoint);",
                          "        }",
                          f"        tinymind::{ty} layer{i};",
                          f"        layer{i}.requantizers = rq{i};"]
            else:
                lines += [f"        tinymind::{ty} layer{i};",
                          f"        layer{i}.requantizer = makeRq({k}_Multiplier, {k}_Shift, {k}_OutputZeroPoint);"]
            bias = f"{k}_Biases" if layer.bias is not None else "nullptr"
            lines += [f"        layer{i}.weights = {k}_Weights;",
                      f"        layer{i}.biases = {bias};",
                      f"        layer{i}.input_zero_point = static_cast<int8_t>({k}_InputZeroPoint);",
                      f"        layer{i}.forward({x}, buf{i});"]
        elif layer.op_type == "MaxPool":
            lines += [f"        tinymind::QMaxPool2D<int8_t, {g['InputHeight']}, {g['InputWidth']}, "
                      f"{g['InputChannels']}, {g['KernelHeight']}, {g['KernelWidth']}, {g['StrideH']}, "
                      f"{g['StrideW']}> layer{i};",
                      f"        layer{i}.forward({x}, buf{i});"]
        elif layer.op_type == "Add":
            lines += [f"        const tinymind::QAddParams p{i} = "
                      f"tinymind::buildQAddParams({k}_InputScale, {k}_InputBScale, {k}_OutputScale);",
                      f"        tinymind::QAdd<int8_t, int8_t, int8_t, {size}> layer{i};",
                      f"        layer{i}.input_a_zero_point = static_cast<int8_t>({k}_InputZeroPoint);",
                      f"        layer{i}.input_b_zero_point = static_cast<int8_t>({k}_InputBZeroPoint);",
                      f"        layer{i}.left_shift = p{i}.left_shift;",
                      f"        layer{i}.input_a_multiplier = p{i}.input_a_multiplier;",
                      f"        layer{i}.input_a_shift = p{i}.input_a_shift;",
                      f"        layer{i}.input_b_multiplier = p{i}.input_b_multiplier;",
                      f"        layer{i}.input_b_shift = p{i}.input_b_shift;",
                      f"        layer{i}.output_requantizer = makeRq(p{i}.output_multiplier, p{i}.output_shift, "
                      f"{k}_OutputZeroPoint);",
                      f"        layer{i}.forward({x}, {buf(layer.sources[1])}, buf{i});"]
        elif layer.op_type in ("Sigmoid", "Tanh"):
            lines += [f"        static int8_t lut{i}[256];",
                      f"        tinymind::buildQ{layer.op_type}LUT({k}_InputScale, {k}_InputZeroPoint, "
                      f"{k}_OutputScale, {k}_OutputZeroPoint, lut{i});",
                      f"        for (std::size_t j = 0; j < {size}; ++j)",
                      "        {",
                      f"            buf{i}[j] = tinymind::qApplyLUT({x}[j], lut{i});",
                      "        }"]
        elif layer.op_type == "Relu":
            lines += [f"        for (std::size_t j = 0; j < {size}; ++j)",
                      "        {",
                      f"            buf{i}[j] = tinymind::qrelu<int8_t>({x}[j], static_cast<int8_t>({k}_InputZeroPoint));",
                      "        }"]
        elif layer.op_type == "Softmax":
            lines += [f"        int32_t multiplier{i} = 0, leftShift{i} = 0, diffMin{i} = 0;",
                      f"        tinymind::buildQSoftmaxFixedPointParams({k}_InputScale, 1.0, "
                      f"multiplier{i}, leftShift{i}, diffMin{i});",
                      f"        const tinymind::QSoftmaxFixedPoint1D<1, {size}> layer{i} = "
                      f"{{multiplier{i}, leftShift{i}, diffMin{i}}};",
                      f"        layer{i}.forward({x}, buf{i});"]
        else:
            raise ValueError(f"{layer.name}: {layer.op_type} has no C++ wiring here")
    last = len(layers) - 1
    lines += [f"        for (std::size_t j = 0; j < {sizes[last]}; ++j)",
              "        {",
              f"            output[j] = buf{last}[j];",
              "        }",
              "    }",
              f"    constexpr std::size_t OutputSize = {sizes[last]};",
              f"    constexpr int32_t InputZeroPoint = {ns}::k{cpp.sanitize(first.name)}_InputZeroPoint;",
              f"    constexpr float InputScale = {ns}::k{cpp.sanitize(first.name)}_InputScale;",
              "}",
              ""]
    with open(os.path.join(out_dir, f"{ns}_net.hpp"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS) + sorted(VARIANTS), default="cnn")
    ap.add_argument("--rust", help="write the Rust module")
    ap.add_argument("--cpp", help="write tinymind (C++)'s header and the C++ network into this directory")
    args = ap.parse_args()
    if not args.rust and not args.cpp:
        ap.error("give --rust and/or --cpp")
    if args.model in VARIANTS:
        model, opts, _ = VARIANTS[args.model]
        what = ("int4 weights in every dense and convolution layer" if "int4_dense" in opts
                else "int16 activations with its int8 weights (16x8)")
        if args.rust:
            layers = ti.import_onnx_model(model_path(model), args.rust, f"onnx_{args.model}",
                                          f"The importer's example {model} with {what}:\n" + MODELS[model],
                                          **opts)
            print(f"wrote {args.rust}: {', '.join(layer.op_type for layer in layers)}")
        if args.cpp:
            write_cpp_variant(args.model, args.cpp)
            print(f"wrote {args.cpp}/onnx_{args.model}.hpp")
        return
    if args.rust:
        layers = ti.import_onnx_model(model_path(args.model), args.rust, f"onnx_{args.model}",
                                      "The importer's example " + MODELS[args.model])
        print(f"wrote {args.rust}: {', '.join(layer.op_type for layer in layers)}")
    if args.cpp:
        write_cpp(args.model, args.cpp)
        print(f"wrote {args.cpp}/onnx_{args.model}_weights.hpp and onnx_{args.model}_net.hpp")


if __name__ == "__main__":
    main()
