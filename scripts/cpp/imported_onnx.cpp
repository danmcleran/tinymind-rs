// The ONNX importer's example models (tools/import_onnx/models, see
// tools/import_onnx/example_onnx.py) on tinymind's C++ int8 runtime: every
// constant comes from the headers tinymind's own ONNX importer writes
// (onnx_<model>_weights.hpp), wired into a forward pass over tinymind's
// layers by example_onnx.py --cpp (onnx_<model>_net.hpp). It runs 16 inputs
// through each model and prints what examples/imported_onnx.rs prints, so
// scripts/compare-with-cpp.sh can byte-compare the two.
//
// A ReLU that ONNX Runtime fused into a layer is that layer's output range
// (zero point -128 = qmin), so the requantizer's clamp applies it, in tinymind
// and in the Rust module alike.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qsoftmax_fixedpoint.hpp"
#include "onnx_cnn_net.hpp"
#include "onnx_cnn_qop_net.hpp"
#include "onnx_mlp_net.hpp"

static const int kSamples = 16;

// The same generator as examples/common/imported_onnx_driver.rs.
static uint32_t gState = 1;

static uint32_t next()
{
    gState = gState * 1664525u + 1013904223u;
    return gState >> 24;
}

static void fill(int s, int32_t zp, int8_t* x, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint32_t r = next();
        int32_t v;
        if (s % 2 == 0)
        {
            v = static_cast<int32_t>(r) - 128;
        }
        else
        {
            v = zp + static_cast<int32_t>(r % 97) - 48;
            v = v < -128 ? -128 : (v > 127 ? 127 : v);
        }
        x[i] = static_cast<int8_t>(v);
    }
}

template<std::size_t In, std::size_t Out>
static void runModel(const char* name, int32_t zp, float scale, void (*forward)(const int8_t*, int8_t*))
{
    uint32_t bits;
    std::memcpy(&bits, &scale, sizeof(bits));
    std::printf("%s input zero_point=%d scale=0x%08x\n", name, static_cast<int>(zp), static_cast<unsigned>(bits));
    for (int s = 0; s < kSamples; ++s)
    {
        int8_t x[In];
        int8_t y[Out];
        fill(s, zp, x, In);
        forward(x, y);
        std::printf("%s %d:", name, s);
        for (std::size_t j = 0; j < Out; ++j)
        {
            std::printf(" %d", static_cast<int>(y[j]));
        }
        std::printf("\n");
    }
}

int main()
{
    runModel<onnx_cnn_net::InputSize, onnx_cnn_net::OutputSize>(
        "onnx_cnn", onnx_cnn_net::InputZeroPoint, onnx_cnn_net::InputScale, onnx_cnn_net::forward);
    runModel<onnx_cnn_qop_net::InputSize, onnx_cnn_qop_net::OutputSize>(
        "onnx_cnn_qop", onnx_cnn_qop_net::InputZeroPoint, onnx_cnn_qop_net::InputScale, onnx_cnn_qop_net::forward);
    runModel<onnx_mlp_net::InputSize, onnx_mlp_net::OutputSize>(
        "onnx_mlp", onnx_mlp_net::InputZeroPoint, onnx_mlp_net::InputScale, onnx_mlp_net::forward);
    return 0;
}
