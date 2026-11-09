// The importer's convolutional example models (tools/import_pytorch/example_cnn.py)
// on tinymind's C++ int8 runtime, from the headers tinymind's own importer
// writes (cnn.hpp, cnn_flat.hpp). It builds the requantizers, the softmax
// parameters and the tanh table at startup, runs 16 inputs through each model,
// and prints what examples/imported_cnn.rs prints, so scripts/compare-with-cpp.sh
// can byte-compare the two.
//
// A ReLU here is qreluBuffer after the requantizer; the Rust module folds it
// into the requantizer's lower bound, which gives the same values. Requantizers
// are printed without their bounds for that reason.
#include <cstdint>
#include <cstdio>

#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qpool2d.hpp"
#include "qsoftmax_fixedpoint.hpp"
#include "cnn.hpp"
#include "cnn_flat.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

static void printRq(const char* model, const char* name, const Rq& rq)
{
    std::printf("%s %s rq=%d,%d,%d\n", model, name, static_cast<int>(rq.multiplier),
                static_cast<int>(rq.shift), static_cast<int>(rq.zero_point));
}

// FNV-1a over int8 tensors, continuing from h.
template<std::size_t N>
static uint32_t fnv(uint32_t h, const int8_t (&t)[N])
{
    for (const int8_t v : t)
    {
        h ^= static_cast<uint8_t>(v);
        h *= 0x01000193u;
    }
    return h;
}

static Rq rq(float in, float w, float out, int32_t zp)
{
    return tinymind::buildRequantizer<int8_t>(in, w, out, zp, -128, 127);
}

// Test input s, element k of the flat NHWC tensor: an integer formula in float,
// as examples/imported_cnn.rs computes it.
template<std::size_t N>
static void input(int s, int8_t (&x)[N], float scale, int32_t zp)
{
    for (std::size_t k = 0; k < N; ++k)
    {
        const int i = static_cast<int>(k);
        const float v = static_cast<float>((s * 23 + i * 29 + s * i * 7) % 37) / 18.0f - 1.0f;
        x[k] = tinymind::quantize<int8_t>(v, scale, zp, -128, 127);
    }
}

static void runCnn()
{
    namespace m = example_cnn;
    tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, 12, 12, 1, 3, 3, 1, 1, 4, 1, 1, 1, 1> conv1;
    conv1.weights = m::kconv1_Weights;
    conv1.biases = m::kconv1_Biases;
    conv1.input_zero_point = static_cast<int8_t>(m::kinput_ZeroPoint);
    conv1.requantizer = rq(m::kinput_Scale, m::kconv1_WeightScale, m::krelu1_Scale, m::krelu1_ZeroPoint);
    const tinymind::QMaxPool2D<int8_t, 12, 12, 4, 2, 2, 2, 2> pool1 = {};
    Rq rq2[4];
    for (Rq& r : rq2)
    {
        r = rq(m::krelu1_Scale, m::kconv2_WeightScale, m::krelu2_Scale, m::krelu2_ZeroPoint);
    }
    tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 6, 6, 4, 3, 3, 1, 1, 1, 1, 1, 1> conv2;
    conv2.weights = m::kconv2_Weights;
    conv2.biases = m::kconv2_Biases;
    conv2.input_zero_point = static_cast<int8_t>(m::krelu1_ZeroPoint);
    conv2.requantizers = rq2;
    tinymind::QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, 6, 6, 4, 8> conv3;
    conv3.weights = m::kconv3_Weights;
    conv3.biases = m::kconv3_Biases;
    conv3.input_zero_point = static_cast<int8_t>(m::krelu2_ZeroPoint);
    conv3.requantizer = rq(m::krelu2_Scale, m::kconv3_WeightScale, m::krelu3_Scale, m::krelu3_ZeroPoint);
    tinymind::QGlobalAvgPool2D<int8_t, int32_t, 6, 6, 8> gap;
    gap.qmin = -128;
    gap.qmax = 127;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 8, 3> fc1;
    fc1.weights = m::kfc1_Weights;
    fc1.biases = m::kfc1_Biases;
    fc1.input_zero_point = static_cast<int8_t>(m::krelu3_ZeroPoint);
    fc1.requantizer = rq(m::krelu3_Scale, m::kfc1_WeightScale, m::kfc1_Scale, m::kfc1_ZeroPoint);
    int32_t multiplier = 0;
    int32_t leftShift = 0;
    int32_t diffMin = 0;
    tinymind::buildQSoftmaxFixedPointParams(m::kfc1_Scale, 1.0, multiplier, leftShift, diffMin);
    const tinymind::QSoftmaxFixedPoint1D<1, 3> softmax = {multiplier, leftShift, diffMin};

    printRq("cnn", "conv1", conv1.requantizer);
    printRq("cnn", "conv2", rq2[0]);
    printRq("cnn", "conv3", conv3.requantizer);
    printRq("cnn", "fc1", fc1.requantizer);
    std::printf("cnn softmax=%d,%d,%d\n", static_cast<int>(multiplier), static_cast<int>(leftShift),
                static_cast<int>(diffMin));
    for (int s = 0; s < 16; ++s)
    {
        int8_t x[144];
        input(s, x, m::kinput_Scale, m::kinput_ZeroPoint);
        int8_t a1[576];
        int8_t p1[144];
        int8_t a2[144];
        int8_t a3[288];
        int8_t g[8];
        int8_t logits[3];
        int8_t probs[3];
        conv1.forward(x, a1);
        tinymind::qreluBuffer(a1, 576, static_cast<int8_t>(m::krelu1_ZeroPoint));
        pool1.forward(a1, p1);
        conv2.forward(p1, a2);
        tinymind::qreluBuffer(a2, 144, static_cast<int8_t>(m::krelu2_ZeroPoint));
        conv3.forward(a2, a3);
        tinymind::qreluBuffer(a3, 288, static_cast<int8_t>(m::krelu3_ZeroPoint));
        gap.forward(a3, g);
        fc1.forward(g, logits);
        softmax.forward(logits, probs);
        int best = 0;
        for (int k = 1; k < 3; ++k)
        {
            if (probs[k] > probs[best])
            {
                best = k;
            }
        }
        // Every intermediate tensor, so each layer is compared, not only the
        // output.
        uint32_t h = 0x811C9DC5u;
        h = fnv(fnv(fnv(fnv(fnv(h, a1), p1), a2), a3), g);
        std::printf("cnn s=%d layers=0x%08X probs=%d,%d,%d class=%d\n", s, static_cast<unsigned>(h),
                    probs[0], probs[1], probs[2], best);
    }
}

static void runCnnFlat()
{
    namespace m = example_cnn_flat;
    tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, 9, 11, 2, 3, 2, 2, 1, 5> conv1;
    conv1.weights = m::kconv1_Weights;
    conv1.biases = m::kconv1_Biases;
    conv1.input_zero_point = static_cast<int8_t>(m::kinput_ZeroPoint);
    conv1.requantizer = rq(m::kinput_Scale, m::kconv1_WeightScale, m::krelu1_Scale, m::krelu1_ZeroPoint);
    tinymind::QAvgPool2D<int8_t, int32_t, 4, 10, 5, 2, 2, 2, 2> pool1;
    pool1.qmin = -128;
    pool1.qmax = 127;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 50, 6> fc1;
    fc1.weights = m::kfc1_Weights;
    fc1.biases = m::kfc1_Biases;
    fc1.input_zero_point = static_cast<int8_t>(m::krelu1_ZeroPoint);
    fc1.requantizer = rq(m::krelu1_Scale, m::kfc1_WeightScale, m::krelu2_Scale, m::krelu2_ZeroPoint);
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 6, 2> fc2;
    fc2.weights = m::kfc2_Weights;
    fc2.biases = m::kfc2_Biases;
    fc2.input_zero_point = static_cast<int8_t>(m::krelu2_ZeroPoint);
    fc2.requantizer = rq(m::krelu2_Scale, m::kfc2_WeightScale, m::kfc2_Scale, m::kfc2_ZeroPoint);
    int8_t lut[tinymind::kQActivationLUTSize];
    tinymind::buildQTanhLUT(m::kfc2_Scale, m::kfc2_ZeroPoint, m::kout_Scale, m::kout_ZeroPoint, lut);

    printRq("cnn_flat", "conv1", conv1.requantizer);
    printRq("cnn_flat", "fc1", fc1.requantizer);
    printRq("cnn_flat", "fc2", fc2.requantizer);
    uint32_t h = 0x811C9DC5u;
    for (const int8_t v : lut)
    {
        h ^= static_cast<uint8_t>(v);
        h *= 0x01000193u;
    }
    std::printf("cnn_flat lut fnv=0x%08X\n", h);
    for (int s = 0; s < 16; ++s)
    {
        int8_t x[198];
        input(s, x, m::kinput_Scale, m::kinput_ZeroPoint);
        int8_t a1[200];
        int8_t p1[50];
        int8_t h1[6];
        int8_t lg[2];
        conv1.forward(x, a1);
        tinymind::qreluBuffer(a1, 200, static_cast<int8_t>(m::krelu1_ZeroPoint));
        pool1.forward(a1, p1);
        fc1.forward(p1, h1);
        tinymind::qreluBuffer(h1, 6, static_cast<int8_t>(m::krelu2_ZeroPoint));
        fc2.forward(h1, lg);
        uint32_t h = 0x811C9DC5u;
        h = fnv(fnv(fnv(fnv(h, a1), p1), h1), lg);
        std::printf("cnn_flat s=%d layers=0x%08X out=%d,%d\n", s, static_cast<unsigned>(h),
                    tinymind::qApplyLUT(lg[0], lut), tinymind::qApplyLUT(lg[1], lut));
    }
}

int main()
{
    runCnn();
    runCnnFlat();
    return 0;
}
