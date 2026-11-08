// tinymind's int8 convolution and pooling layers over random tensors:
// QConv2D and QConv2DPerChannel, QDepthwiseConv2D, QPointwiseConv2D and
// QPointwiseConv2DPerChannel, QMaxPool2D, QAvgPool2D and QGlobalAvgPool2D.
//
// The shapes cover each code path of the portable (non-SIMD) build: convolution windows gathered into
// patches (up to 64 bytes) and read row by row (above), output rows split into
// several tiles, filters in pairs with an odd one left over, odd output widths,
// strides above the kernel, padding on every side (asymmetric, and as wide as
// the kernel, so some windows lie wholly in the padding) and single-pixel
// pointwise layers. Requantizers mix the fused path (shift 2 to 31) with the
// two-step one (shift below 2), the per-filter fast form with the general one
// (a fused ReLU range, a narrow range, a large zero point at a large shift),
// and biases are sometimes absent. Pooling ranges include qmin > qmax.
//
// Each layer runs TRIALS times; its line hashes every output byte. The draws
// come in a fixed order (input zero point, input, weights, biases, then the
// requantizers), which tests/conv_sweep.rs repeats with the Rust layers.
// tests/golden/conv/stdout.txt is this program's output.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qpool2d.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

constexpr int TRIALS = 3;

static uint32_t gState = 0x5EED0001u;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static uint32_t draw(const uint32_t n)
{
    return (nextU32() >> 8) % n;
}

static int8_t randI8()
{
    return static_cast<int8_t>(static_cast<int32_t>(draw(256)) - 128);
}

static std::vector<int8_t> randTensor(const std::size_t n)
{
    std::vector<int8_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i] = randI8();
    }
    return v;
}

// Biases are always drawn, so the draw order does not depend on whether the
// layer uses them; the caller passes nullptr for a quarter of the trials.
static std::vector<int32_t> randBiases(const std::size_t n, bool& use)
{
    use = draw(4) != 0;
    std::vector<int32_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i] = static_cast<int32_t>(draw(1u << 17)) - (1 << 16);
    }
    return v;
}

static Rq randRq()
{
    Rq rq;
    rq.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
    const uint32_t kind = draw(8);
    if (kind == 0)
    {
        rq.shift = static_cast<int32_t>(draw(6)) - 3; // -3..2: the two-step path below 2
    }
    else if (kind == 1)
    {
        rq.shift = 20 + static_cast<int32_t>(draw(12)); // 20..31: large zero-point terms
    }
    else
    {
        rq.shift = 6 + static_cast<int32_t>(draw(8)); // 6..13: outputs spread over the range
    }
    rq.zero_point = randI8();
    const uint32_t range = draw(4);
    if (range == 0)
    {
        rq.qmin = rq.zero_point; // fused ReLU
        rq.qmax = 127;
    }
    else if (range == 1)
    {
        rq.qmin = -100;
        rq.qmax = 90;
    }
    else
    {
        rq.qmin = -128;
        rq.qmax = 127;
    }
    return rq;
}

static std::vector<Rq> randRqs(const std::size_t n)
{
    std::vector<Rq> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i] = randRq();
    }
    return v;
}

static void randRange(int8_t& qmin, int8_t& qmax)
{
    const uint32_t kind = draw(3);
    if (kind == 0)
    {
        qmin = -128;
        qmax = 127;
    }
    else if (kind == 1)
    {
        qmin = -20;
        qmax = 20;
    }
    else
    {
        qmin = randI8(); // may exceed qmax: the result is then qmax
        qmax = randI8();
    }
}

static uint32_t gHash;

static void hashBytes(const int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        gHash ^= static_cast<uint8_t>(p[i]);
        gHash *= 0x01000193u;
    }
}

static void report(const char* kind, const char* name)
{
    std::printf("%s %s 0x%08X\n", kind, name, static_cast<unsigned>(gHash));
}

template<typename Layer, bool PerChannel>
static void conv(const char* kind, const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < TRIALS; ++t)
    {
        const int8_t zp = randI8();
        const std::vector<int8_t> input = randTensor(Layer::InputSize);
        const std::vector<int8_t> weights = randTensor(Layer::TotalWeights);
        bool useBiases = false;
        const std::vector<int32_t> biases = randBiases(Layer::NumFilters, useBiases);
        const std::vector<Rq> rqs = randRqs(PerChannel ? Layer::NumFilters : 1);
        std::vector<int8_t> output(Layer::OutputSize);

        Layer layer;
        layer.weights = weights.data();
        layer.biases = useBiases ? biases.data() : nullptr;
        layer.input_zero_point = zp;
        if constexpr (PerChannel)
        {
            layer.requantizers = rqs.data();
        }
        else
        {
            layer.requantizer = rqs[0];
        }
        layer.forward(input.data(), output.data());
        hashBytes(output.data(), output.size());
    }
    report(kind, name);
}

template<typename Layer>
static void depthwise(const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < TRIALS; ++t)
    {
        const int8_t zp = randI8();
        const std::vector<int8_t> input = randTensor(Layer::InputSize);
        const std::vector<int8_t> weights = randTensor(Layer::TotalWeights);
        bool useBiases = false;
        const std::vector<int32_t> biases = randBiases(Layer::Channels, useBiases);
        const std::vector<Rq> rqs = randRqs(Layer::Channels);
        std::vector<int8_t> output(Layer::OutputSize);

        Layer layer;
        layer.weights = weights.data();
        layer.biases = useBiases ? biases.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizers = rqs.data();
        layer.forward(input.data(), output.data());
        hashBytes(output.data(), output.size());
    }
    report("depthwise", name);
}

template<typename Layer>
static void maxPool(const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < TRIALS; ++t)
    {
        const std::vector<int8_t> input = randTensor(Layer::InputSize);
        std::vector<int8_t> output(Layer::OutputSize);
        Layer layer;
        layer.forward(input.data(), output.data());
        hashBytes(output.data(), output.size());
    }
    report("max_pool", name);
}

template<typename Layer>
static void avgPool(const char* kind, const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < TRIALS; ++t)
    {
        const std::vector<int8_t> input = randTensor(Layer::InputSize);
        std::vector<int8_t> output(Layer::OutputSize);
        Layer layer;
        randRange(layer.qmin, layer.qmax);
        layer.forward(input.data(), output.data());
        hashBytes(output.data(), output.size());
    }
    report(kind, name);
}

#define CONV(name, H, W, C, KH, KW, SH, SW, F, PT, PB, PL, PR)                                        \
    conv<tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL, PR>, \
         false>("conv", name);                                                                              \
    conv<tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT,   \
                                     PB, PL, PR>,                                                   \
         true>("conv_per_channel", name)

#define DEPTHWISE(name, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR) \
    depthwise<tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR>>(name)

#define POINTWISE(name, H, W, C, F)                                                          \
    conv<tinymind::QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, F>, false>("pointwise", name); \
    conv<tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, F>, true>("pointwise_per_channel", name)

#define POOL(name, H, W, C, PH, PW, SH, SW)                                    \
    maxPool<tinymind::QMaxPool2D<int8_t, H, W, C, PH, PW, SH, SW>>(name);      \
    avgPool<tinymind::QAvgPool2D<int8_t, int32_t, H, W, C, PH, PW, SH, SW>>("avg_pool", name)

#define GLOBAL_AVG_POOL(name, H, W, C) \
    avgPool<tinymind::QGlobalAvgPool2D<int8_t, int32_t, H, W, C>>("global_avg_pool", name)

int main()
{
    //   name             H   W   C  KH KW SH SW  F PT PB PL PR
    CONV("stem_20x20",   20, 20,  1, 3, 3, 1, 1, 8, 0, 0, 0, 0);
    CONV("same_3x3",      7,  9,  3, 3, 3, 1, 1, 5, 1, 1, 1, 1);
    CONV("rows_3x3x8",   10, 10,  8, 3, 3, 1, 1, 4, 0, 0, 0, 0);
    CONV("rows_stride2",  9, 11,  8, 3, 3, 2, 2, 7, 1, 1, 1, 1);
    CONV("asym_pad",     12,  5,  2, 4, 2, 3, 1, 3, 0, 2, 1, 0);
    CONV("one_by_one",    6,  6,  1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    CONV("dscnn_stem",   49, 10,  1,10, 4, 2, 2, 8, 4, 5, 1, 2);
    CONV("tiles_1x5",     5, 40,  2, 1, 5, 1, 1, 6, 0, 0, 2, 2);
    CONV("rows_padded",   3,  3, 16, 3, 3, 1, 1, 2, 1, 1, 1, 1);
    CONV("stride_4x3",    8,  8,  4, 5, 5, 4, 3, 9, 0, 0, 0, 0);
    CONV("stride_gt_k",   7,  7,  1, 2, 2, 3, 3, 2, 0, 1, 0, 1);
    CONV("wide_pad",      4,  4,  2, 3, 3, 1, 1, 3, 2, 2, 2, 2);
    CONV("all_pad",       2,  2,  1, 2, 2, 1, 1, 2, 2, 2, 2, 2);

    //        name             H   W   C KH KW SH SW PT PB PL PR
    DEPTHWISE("kws_3x3",       9,  9,  8, 3, 3, 1, 1, 0, 0, 0, 0);
    DEPTHWISE("same_3x3",     13,  5, 12, 3, 3, 1, 1, 1, 1, 1, 1);
    DEPTHWISE("asym_pad",      8,  7,  3, 3, 2, 2, 1, 0, 2, 1, 0);
    DEPTHWISE("big_kernel",    6,  6,  5, 5, 5, 1, 1, 0, 0, 0, 0);
    DEPTHWISE("one_by_one",   10,  4,  2, 1, 1, 3, 2, 0, 0, 0, 0);
    DEPTHWISE("stride_gt_k",   7,  7,  4, 2, 2, 3, 3, 1, 0, 0, 1);
    DEPTHWISE("all_pad",       2,  2,  1, 2, 2, 1, 1, 2, 2, 2, 2);

    //        name          H  W   C   F
    POINTWISE("kws_8_16",   7, 7,  8, 16);
    POINTWISE("pixel",      1, 1, 16, 10);
    POINTWISE("wide",       5, 3, 40,  6);
    POINTWISE("odd",        3, 3, 33,  5);
    POINTWISE("tiny",       4, 1,  1,  1);

    //   name            H   W  C PH PW SH SW
    POOL("kws_2x2",     18, 18, 8, 2, 2, 2, 2);
    POOL("overlap_3x3",  7,  9, 3, 3, 3, 2, 2);
    POOL("rect_2x3",     5,  5, 2, 2, 3, 1, 2);
    POOL("big_window",   6,  4, 1, 4, 4, 1, 1);
    POOL("tiled_3x3",    9,  9, 4, 3, 3, 3, 3);

    //              name      H  W   C
    GLOBAL_AVG_POOL("kws",    7, 7, 16);
    GLOBAL_AVG_POOL("pixel",  1, 1,  3);
    GLOBAL_AVG_POOL("rect",   5, 3, 10);
    GLOBAL_AVG_POOL("odd",   25, 5,  8);
    return 0;
}
