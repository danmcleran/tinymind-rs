// tinymind's 16x8 paths (int16 activations, int8 weights, int64
// accumulators; cpp/qaffine.hpp, qdense.hpp, qconv2d.hpp,
// qdepthwiseconv2d.hpp, qpool2d.hpp, qadd.hpp, qactivations.hpp,
// qsoftmax_fixedpoint.hpp) over random parameters and tensors:
//
//   rq64      Requantizer<int64_t, int16_t>::apply on accumulators across and
//             beyond 2^47, multipliers including negative ones and ones at or
//             above 0x7FFF0000, shifts from -20 to 50 (outside TFLite's
//             -7..31 contract too), zero points and ranges with qmin > qmax
//   dense     QDense (one requantizer) on odd and even sizes
//   conv      QConv2D and QConv2DPerChannel with padding and strides
//   depthwise QDepthwiseConv2D, with windows of up to 256 taps (summed in
//             int32 by the C++) and above (int64)
//   maxpool   QMaxPool2D
//   add       QAdd<int16_t> from buildQAddParams(..., 15) and with random
//             parameters (the int32 requantizer onto int16)
//   lut       qApplyLUT16 over all 65,536 inputs on random tables and on
//             buildQSigmoidLUT16 / buildQTanhLUT16 tables
//   softmax   QSoftmaxInt16 from buildQSoftmaxInt16Params, rows of 1 to 300
//
// Each line hashes its outputs (FNV-1a over little-endian int16).
// tests/golden/int16x8/stdout.txt is this program's output, and
// tests/int16x8_sweep.rs computes the same in Rust.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#define TINYMIND_ENABLE_FLOAT 1
#define TINYMIND_ENABLE_STD 1
#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qadd.hpp"
#include "qcalibration.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpool2d.hpp"
#include "qsoftmax_fixedpoint.hpp"

typedef tinymind::Requantizer<int64_t, int16_t> Rq;

static uint32_t gState = 0x16081608u;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static uint32_t draw(uint32_t n)
{
    return (nextU32() >> 8) % n;
}

static int16_t randI16()
{
    return static_cast<int16_t>(static_cast<uint16_t>(nextU32() >> 16));
}

static int8_t randI8()
{
    return static_cast<int8_t>(static_cast<int32_t>(draw(256)) - 128);
}

static int64_t randI64()
{
    const uint64_t high = nextU32() >> 16;
    const uint64_t mid = nextU32() >> 16;
    const uint64_t low = nextU32() >> 16;
    return static_cast<int64_t>((high << 48) | (mid << 32) | (low << 16) | (nextU32() >> 16));
}

// 1.xxx * 2^-e, the mantissa drawn before the exponent (one draw per
// statement: the order of two calls in one expression is unspecified).
static double randScale(uint32_t minExp, uint32_t spread)
{
    const double mantissa = 1.0 + static_cast<double>(draw(1000)) / 1000.0;
    const int e = static_cast<int>(minExp + draw(spread));
    return std::ldexp(mantissa, -e);
}

static uint32_t gHash;

static void hash16(int16_t v)
{
    const uint16_t u = static_cast<uint16_t>(v);
    gHash = (gHash ^ (u & 0xFFu)) * 0x01000193u;
    gHash = (gHash ^ (u >> 8)) * 0x01000193u;
}

static void report(const char* kind, const char* name)
{
    std::printf("%s %s 0x%08X\n", kind, name, static_cast<unsigned>(gHash));
}

// A requantizer as the 16x8 layers use it: a scale ratio below 1 (shift 0 to
// 14), a zero point (TFLite: 0) and the full range or a narrower one.
static Rq randRq()
{
    Rq rq;
    rq.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
    rq.shift = static_cast<int32_t>(draw(15));
    rq.zero_point = (draw(2) == 0) ? static_cast<int16_t>(0) : static_cast<int16_t>(randI16() / 4);
    const uint32_t range = draw(3);
    rq.qmin = (range == 0) ? rq.zero_point : static_cast<int16_t>(range == 1 ? -20000 : -32768);
    rq.qmax = (range == 1) ? static_cast<int16_t>(25000) : static_cast<int16_t>(32767);
    return rq;
}

// A fine requantizer (a ratio near 1, zero point 0, the full range) for the
// trials with small inputs, where an error of one in a product shows.
static Rq fineRq()
{
    Rq rq;
    rq.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
    rq.shift = static_cast<int32_t>(draw(4));
    rq.zero_point = 0;
    rq.qmin = -32768;
    rq.qmax = 32767;
    return rq;
}

// Trial 1 of each layer runs small inputs (-64..63) through fine
// requantizers; the others run full-range inputs.
static int16_t trialInput(int t)
{
    const int16_t v = randI16();
    return (t == 1) ? static_cast<int16_t>(v >> 9) : v;
}

// Every corner of the requantizer.
static Rq anyRq()
{
    Rq rq;
    const uint32_t m = draw(6);
    rq.multiplier = (m == 0) ? static_cast<int32_t>(nextU32())
                  : (m == 1) ? static_cast<int32_t>(0x7FFF0000 + draw(0x10000))
                  : (m == 2) ? -static_cast<int32_t>(draw(1u << 30))
                  : static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
    rq.shift = static_cast<int32_t>(draw(71)) - 20;
    rq.zero_point = randI16();
    rq.qmin = randI16();
    rq.qmax = (draw(4) == 0) ? randI16() : static_cast<int16_t>(32767);
    if (draw(3) == 0)
    {
        rq.qmin = -32768;
    }
    return rq;
}

static void requantizers()
{
    gHash = 0x811C9DC5u;
    for (int i = 0; i < 20000; ++i)
    {
        const Rq rq = anyRq();
        const uint32_t kind = draw(4);
        int64_t acc = randI64();
        if (kind == 0)
        {
            acc >>= 16; // within 2^47
        }
        else if (kind == 1)
        {
            acc >>= 40;
        }
        else if (kind == 2)
        {
            const int64_t v = static_cast<int64_t>(randI16());
            acc = v * (static_cast<int64_t>(1) << draw(40));
        }
        hash16(rq.apply(acc));
    }
    report("rq64", "corners");
}

template<std::size_t In, std::size_t Out>
static void dense(const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < 4; ++t)
    {
        const int16_t zp = (t == 0) ? static_cast<int16_t>(0) : randI16();
        std::vector<int16_t> x(In);
        for (auto& v : x) v = trialInput(t);
        std::vector<int8_t> w(In * Out);
        for (auto& v : w) v = randI8();
        const bool useBias = draw(4) != 0;
        std::vector<int64_t> b(Out);
        for (auto& v : b) v = static_cast<int64_t>(randI64() >> 30);
        tinymind::QDense<int16_t, int8_t, int64_t, int16_t, In, Out> layer;
        layer.weights = w.data();
        layer.biases = useBias ? b.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizer = (t == 1) ? fineRq() : randRq();
        int16_t y[Out];
        layer.forward(x.data(), y);
        for (std::size_t i = 0; i < Out; ++i) hash16(y[i]);
    }
    report("dense", name);
}

template<bool PerChannel, std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW,
         std::size_t SH, std::size_t SW, std::size_t F, std::size_t PT, std::size_t PB, std::size_t PL,
         std::size_t PR>
static void conv(const char* name)
{
    typedef tinymind::QConv2D<int16_t, int8_t, int64_t, int16_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL, PR> T1;
    typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL,
                                        PR> TN;
    gHash = 0x811C9DC5u;
    for (int t = 0; t < 3; ++t)
    {
        const int16_t zp = (t == 0) ? static_cast<int16_t>(0) : randI16();
        std::vector<int16_t> x(H * W * C);
        for (auto& v : x) v = trialInput(t);
        std::vector<int8_t> w(F * KH * KW * C);
        for (auto& v : w) v = randI8();
        const bool useBias = draw(4) != 0;
        std::vector<int64_t> b(F);
        for (auto& v : b) v = static_cast<int64_t>(randI64() >> 30);
        std::vector<Rq> rqs(F);
        for (auto& r : rqs) r = (t == 1) ? fineRq() : randRq();
        std::vector<int16_t> y(T1::OutputSize);
        if (PerChannel)
        {
            TN layer;
            layer.weights = w.data();
            layer.biases = useBias ? b.data() : nullptr;
            layer.input_zero_point = zp;
            layer.requantizers = rqs.data();
            layer.forward(x.data(), y.data());
        }
        else
        {
            T1 layer;
            layer.weights = w.data();
            layer.biases = useBias ? b.data() : nullptr;
            layer.input_zero_point = zp;
            layer.requantizer = rqs[0];
            layer.forward(x.data(), y.data());
        }
        for (int16_t v : y) hash16(v);
    }
    report(PerChannel ? "conv-per-channel" : "conv", name);
}

template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW, std::size_t SH,
         std::size_t SW, std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
static void depthwise(const char* name)
{
    typedef tinymind::QDepthwiseConv2D<int16_t, int8_t, int64_t, int16_t, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR>
        Layer;
    gHash = 0x811C9DC5u;
    for (int t = 0; t < 3; ++t)
    {
        const int16_t zp = (t == 0) ? static_cast<int16_t>(0) : randI16();
        std::vector<int16_t> x(H * W * C);
        for (auto& v : x) v = trialInput(t);
        std::vector<int8_t> w(C * KH * KW);
        for (auto& v : w) v = randI8();
        const bool useBias = draw(4) != 0;
        std::vector<int64_t> b(C);
        for (auto& v : b) v = static_cast<int64_t>(randI64() >> 30);
        std::vector<Rq> rqs(C);
        for (auto& r : rqs) r = (t == 1) ? fineRq() : randRq();
        Layer layer;
        layer.weights = w.data();
        layer.biases = useBias ? b.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizers = rqs.data();
        std::vector<int16_t> y(Layer::OutputSize);
        layer.forward(x.data(), y.data());
        for (int16_t v : y) hash16(v);
    }
    report("depthwise", name);
}

template<std::size_t H, std::size_t W, std::size_t C, std::size_t PH, std::size_t PW, std::size_t SH, std::size_t SW>
static void maxPool(const char* name)
{
    typedef tinymind::QMaxPool2D<int16_t, H, W, C, PH, PW, SH, SW> Layer;
    gHash = 0x811C9DC5u;
    std::vector<int16_t> x(H * W * C);
    for (auto& v : x) v = randI16();
    std::vector<int16_t> y(Layer::OutputSize);
    Layer layer;
    layer.forward(x.data(), y.data());
    for (int16_t v : y) hash16(v);
    report("maxpool", name);
}

static void adds()
{
    const std::size_t n = 257;
    gHash = 0x811C9DC5u;
    for (int t = 0; t < 24; ++t)
    {
        tinymind::QAdd<int16_t, int16_t, int16_t, n> add;
        if (t < 16)
        {
            // Calibrated as TFLite's int16 ADD: zero points 0, left shift 15.
            const float sa = static_cast<float>(randScale(0, 14));
            const float sb = static_cast<float>(randScale(0, 14));
            const float so = static_cast<float>(randScale(0, 13));
            const tinymind::QAddParams p = tinymind::buildQAddParams(sa, sb, so, 15);
            add.input_a_zero_point = 0;
            add.input_b_zero_point = 0;
            add.left_shift = p.left_shift;
            add.input_a_multiplier = p.input_a_multiplier;
            add.input_a_shift = p.input_a_shift;
            add.input_b_multiplier = p.input_b_multiplier;
            add.input_b_shift = p.input_b_shift;
            add.output_requantizer.multiplier = p.output_multiplier;
            add.output_requantizer.shift = p.output_shift;
            add.output_requantizer.zero_point = 0;
            add.output_requantizer.qmin = -32768;
            add.output_requantizer.qmax = 32767;
            std::printf("add calibrated %d: %d %d %d %d %d %d %d\n", t, p.left_shift, p.input_a_multiplier,
                        p.input_a_shift, p.input_b_multiplier, p.input_b_shift, p.output_multiplier, p.output_shift);
        }
        else
        {
            // Random parameters within the C++'s contract (no int32 overflow
            // in the shifted inputs): small zero points, shifts 10 to 15.
            add.input_a_zero_point = static_cast<int16_t>(static_cast<int32_t>(draw(201)) - 100);
            add.input_b_zero_point = static_cast<int16_t>(static_cast<int32_t>(draw(201)) - 100);
            add.left_shift = 10 + static_cast<int32_t>(draw(6));
            add.input_a_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
            add.input_a_shift = static_cast<int32_t>(draw(4));
            add.input_b_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
            add.input_b_shift = static_cast<int32_t>(draw(4));
            add.output_requantizer.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
            add.output_requantizer.shift = static_cast<int32_t>(draw(20)) - 2;
            add.output_requantizer.zero_point = static_cast<int16_t>(randI16() / 8);
            add.output_requantizer.qmin = static_cast<int16_t>(-30000 + static_cast<int32_t>(draw(1000)));
            add.output_requantizer.qmax = static_cast<int16_t>(30000 - static_cast<int32_t>(draw(1000)));
        }
        std::vector<int16_t> a(n), b(n), y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            a[i] = (t < 16) ? randI16() : static_cast<int16_t>(randI16() / 2);
            b[i] = (t < 16) ? randI16() : static_cast<int16_t>(randI16() / 2);
        }
        add.forward(a.data(), b.data(), y.data());
        for (int16_t v : y) hash16(v);
    }
    report("add", "int16");
}

static void luts()
{
    int16_t lut[tinymind::kQActivationLUT16Size];
    for (int t = 0; t < 6; ++t)
    {
        gHash = 0x811C9DC5u;
        if (t < 2)
        {
            for (auto& v : lut) v = randI16(); // steep slopes wrap
        }
        else
        {
            const double inScale = randScale(9, 4);
            const int32_t inZp = (t % 2 == 0) ? 0 : static_cast<int32_t>(randI16() / 4);
            if (t < 4)
            {
                tinymind::buildQSigmoidLUT16(inScale, inZp, 1.0 / 32768.0, lut);
            }
            else
            {
                tinymind::buildQTanhLUT16(inScale, inZp, 1.0 / 32768.0, lut);
            }
            uint32_t h = 0x811C9DC5u;
            for (int16_t v : lut)
            {
                const uint16_t u = static_cast<uint16_t>(v);
                h = (h ^ (u & 0xFFu)) * 0x01000193u;
                h = (h ^ (u >> 8)) * 0x01000193u;
            }
            std::printf("lut table %d 0x%08X\n", t, static_cast<unsigned>(h));
        }
        for (int32_t x = -32768; x <= 32767; ++x)
        {
            hash16(tinymind::qApplyLUT16(static_cast<int16_t>(x), lut));
        }
        std::printf("lut apply %d 0x%08X\n", t, static_cast<unsigned>(gHash));
    }
}

template<std::size_t N>
static void softmaxRows(const char* name)
{
    gHash = 0x811C9DC5u;
    for (int t = 0; t < 8; ++t)
    {
        const float scale = static_cast<float>(randScale(10, 6));
        const float beta = (t % 3 == 0) ? 1.0f : 0.5f + static_cast<float>(draw(100)) / 50.0f;
        tinymind::QSoftmaxInt16<1, N> softmax;
        tinymind::buildQSoftmaxInt16Params(static_cast<double>(scale), static_cast<double>(beta),
                                           softmax.input_multiplier, softmax.input_shift);
        std::printf("softmax %s %d params %d %d\n", name, t, softmax.input_multiplier, softmax.input_shift);
        int16_t x[N];
        int16_t y[N];
        for (std::size_t i = 0; i < N; ++i)
        {
            x[i] = (t == 7) ? static_cast<int16_t>(-32768 + 65535 * (i % 2)) : randI16();
        }
        softmax.forward(x, y);
        for (int16_t v : y) hash16(v);
    }
    report("softmax", name);
}

int main()
{
    requantizers();
    dense<1, 1>("1x1");
    dense<7, 5>("7x5");
    dense<64, 10>("64x10");
    dense<300, 3>("300x3");
    conv<false, 5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1>("3x3-pad");
    conv<true, 5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1>("3x3-pad");
    conv<true, 7, 7, 1, 3, 2, 2, 1, 3, 2, 0, 1, 1>("strided-asym");
    conv<true, 4, 4, 9, 1, 1, 1, 1, 5, 0, 0, 0, 0>("pointwise");
    conv<false, 6, 5, 2, 2, 3, 2, 2, 6, 0, 1, 2, 0>("odd");
    depthwise<6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1>("3x3-pad");
    depthwise<8, 8, 4, 5, 5, 2, 1, 2, 2, 0, 1>("5x5-strided");
    depthwise<17, 18, 2, 17, 17, 1, 1, 0, 0, 1, 1>("289-taps");
    maxPool<6, 6, 3, 2, 2, 2, 2>("2x2");
    maxPool<7, 5, 2, 3, 2, 2, 1>("3x2-strided");
    adds();
    luts();
    softmaxRows<1>("1");
    softmaxRows<2>("2");
    softmaxRows<10>("10");
    softmaxRows<300>("300");
    return 0;
}
