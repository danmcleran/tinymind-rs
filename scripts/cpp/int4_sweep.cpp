// tinymind's packed int4 layers (cpp/qint4.hpp) over random tensors:
// QDenseInt4, QConv2DInt4 (layers small enough to unpack their weights, and
// layers above TINYMIND_INT4_CONV_UNPACK_MAX_BYTES that compute from the
// packed weights through a gathered patch, in place for a one-row kernel, and
// row by row), QDepthwiseConv2DInt4, packInt4 / unpackInt4 and
// quantizeInt4PerChannel. Odd windows put every second filter's weights
// mid-byte. Each layer's output is also compared with the int8 layer on the
// unpacked weights (same=1). Requantizers, biases and draws as in
// conv_sweep.cpp. tests/golden/int4/stdout.txt is this program's output, and
// tests/int4_sweep.rs computes the same in Rust.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define TINYMIND_ENABLE_FLOAT 1
#define TINYMIND_ENABLE_STD 1
#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qint4.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

constexpr int TRIALS = 3;

static uint32_t gState = 0x5EED0004u;

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

static uint32_t gHash;

static void hashBytes(const int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        gHash ^= static_cast<uint8_t>(p[i]);
        gHash *= 0x01000193u;
    }
}

static std::vector<uint8_t> randPacked(const std::size_t n)
{
    std::vector<uint8_t> v(tinymind::int4PackedBytes(n));
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        v[i] = static_cast<uint8_t>(draw(256));
    }
    return v;
}

static std::vector<int8_t> unpacked(const std::vector<uint8_t>& packed, const std::size_t n)
{
    std::vector<int8_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i] = tinymind::unpackInt4(packed.data(), i);
    }
    return v;
}

static void report(const char* kind, const char* name, bool same)
{
    std::printf("%s %s 0x%08X same=%d\n", kind, name, static_cast<unsigned>(gHash), same ? 1 : 0);
}

template<std::size_t In, std::size_t Out>
static void dense(const char* name)
{
    gHash = 0x811C9DC5u;
    bool same = true;
    for (int t = 0; t < TRIALS; ++t)
    {
        const int8_t zp = randI8();
        const std::vector<int8_t> x = randTensor(In);
        const std::vector<uint8_t> w = randPacked(In * Out);
        bool useBias = false;
        const std::vector<int32_t> b = randBiases(Out, useBias);
        const std::vector<Rq> rqs = randRqs(Out);
        tinymind::QDenseInt4<int8_t, int32_t, int8_t, In, Out> layer;
        layer.weights = w.data();
        layer.biases = useBias ? b.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizers = rqs.data();
        int8_t y[Out];
        layer.forward(x.data(), y);
        hashBytes(y, Out);
        // The int8 reference: QConv2DPerChannel 1x1 over one pixel.
        const std::vector<int8_t> w8 = unpacked(w, In * Out);
        tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, In, 1, 1, 1, 1, Out> ref;
        ref.weights = w8.data();
        ref.biases = layer.biases;
        ref.input_zero_point = zp;
        ref.requantizers = rqs.data();
        int8_t r[Out];
        ref.forward(x.data(), r);
        same = same && std::memcmp(y, r, Out) == 0;
    }
    report("dense", name, same);
}

template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW, std::size_t SH,
         std::size_t SW, std::size_t F, std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
static void conv(const char* name)
{
    typedef tinymind::QConv2DInt4<int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL, PR> Layer;
    typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL,
                                        PR> Ref;
    gHash = 0x811C9DC5u;
    bool same = true;
    for (int t = 0; t < TRIALS; ++t)
    {
        const int8_t zp = randI8();
        const std::vector<int8_t> x = randTensor(H * W * C);
        const std::vector<uint8_t> w = randPacked(F * KH * KW * C);
        bool useBias = false;
        const std::vector<int32_t> b = randBiases(F, useBias);
        const std::vector<Rq> rqs = randRqs(F);
        Layer layer;
        layer.weights = w.data();
        layer.biases = useBias ? b.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizers = rqs.data();
        std::vector<int8_t> y(Layer::OutputSize);
        layer.forward(x.data(), y.data());
        hashBytes(y.data(), y.size());
        const std::vector<int8_t> w8 = unpacked(w, F * KH * KW * C);
        Ref ref;
        ref.weights = w8.data();
        ref.biases = layer.biases;
        ref.input_zero_point = zp;
        ref.requantizers = rqs.data();
        std::vector<int8_t> r(Layer::OutputSize);
        ref.forward(x.data(), r.data());
        same = same && y == r;
    }
    std::printf("# %s unpacks=%d\n", name, Layer::UnpacksWeights ? 1 : 0);
    report("conv", name, same);
}

template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW, std::size_t SH,
         std::size_t SW, std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
static void depthwise(const char* name)
{
    typedef tinymind::QDepthwiseConv2DInt4<int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR> Layer;
    typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR>
        Ref;
    gHash = 0x811C9DC5u;
    bool same = true;
    for (int t = 0; t < TRIALS; ++t)
    {
        const int8_t zp = randI8();
        const std::vector<int8_t> x = randTensor(H * W * C);
        const std::vector<uint8_t> w = randPacked(C * KH * KW);
        bool useBias = false;
        const std::vector<int32_t> b = randBiases(C, useBias);
        const std::vector<Rq> rqs = randRqs(C);
        Layer layer;
        layer.weights = w.data();
        layer.biases = useBias ? b.data() : nullptr;
        layer.input_zero_point = zp;
        layer.requantizers = rqs.data();
        std::vector<int8_t> y(Layer::OutputSize);
        layer.forward(x.data(), y.data());
        hashBytes(y.data(), y.size());
        const std::vector<int8_t> w8 = unpacked(w, C * KH * KW);
        Ref ref;
        ref.weights = w8.data();
        ref.biases = layer.biases;
        ref.input_zero_point = zp;
        ref.requantizers = rqs.data();
        std::vector<int8_t> r(Layer::OutputSize);
        ref.forward(x.data(), r.data());
        same = same && y == r;
    }
    report("depthwise", name, same);
}

static void packing()
{
    // Every value -8..7 at both nibbles, an odd length, out-of-range values.
    int8_t values[35];
    for (int i = 0; i < 35; ++i)
    {
        values[i] = static_cast<int8_t>((i < 32) ? (i % 16) - 8 : static_cast<int>(randI8()));
    }
    uint8_t packed[18];
    tinymind::packInt4(values, 35, packed);
    std::printf("pack");
    for (int i = 0; i < 18; ++i)
    {
        std::printf(" %02x", packed[i]);
    }
    std::printf("\nunpack");
    for (int i = 0; i < 35; ++i)
    {
        std::printf(" %d", tinymind::unpackInt4(packed, static_cast<std::size_t>(i)));
    }
    std::printf("\n");
}

static void quantize()
{
    const std::size_t channels = 4;
    const std::size_t per = 9;
    float w[channels * per];
    for (std::size_t i = 0; i < channels * per; ++i)
    {
        w[i] = static_cast<float>(static_cast<int32_t>(draw(2001)) - 1000) / 1024.0f;
    }
    for (std::size_t i = 0; i < per; ++i)
    {
        w[2 * per + i] = 0.0f; // a channel of zeros
    }
    // Ties: multiples of half the scale (max 3.5, scale 0.5).
    const float ties[per] = {3.5f, -0.25f, 0.75f, -1.25f, 2.25f, -3.25f, 0.0f, 1.75f, -2.75f};
    std::memcpy(w + 3 * per, ties, sizeof(ties));
    int8_t codes[channels * per];
    float scales[channels];
    tinymind::quantizeInt4PerChannel(w, channels, per, codes, scales);
    for (std::size_t c = 0; c < channels; ++c)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &scales[c], 4);
        std::printf("quantize c=%zu scale=%08x codes=", c, static_cast<unsigned>(bits));
        for (std::size_t i = 0; i < per; ++i)
        {
            std::printf("%s%d", (i == 0) ? "" : ",", codes[c * per + i]);
        }
        std::printf("\n");
    }
}

int main()
{
    dense<1, 1>("1x1");
    dense<3, 2>("3x2");
    dense<7, 5>("7x5");
    dense<8, 4>("8x4");
    dense<33, 10>("33x10");
    dense<64, 16>("64x16");
    conv<5, 6, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1>("unpack-3x3-pad");
    conv<7, 7, 1, 3, 2, 2, 1, 3, 2, 0, 1, 1>("unpack-odd-window");
    conv<4, 4, 5, 1, 1, 1, 1, 7, 0, 0, 0, 0>("unpack-pointwise");
    conv<6, 6, 8, 3, 3, 1, 1, 16, 1, 1, 1, 1>("packed-patch");
    conv<5, 6, 7, 3, 1, 2, 1, 50, 1, 1, 0, 0>("packed-patch-odd");
    conv<5, 6, 15, 3, 3, 2, 1, 8, 1, 1, 1, 0>("packed-rows-odd");
    conv<4, 5, 11, 1, 3, 1, 1, 32, 0, 0, 1, 1>("packed-one-row-odd");
    depthwise<6, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1>("3x3-pad");
    depthwise<8, 8, 40, 5, 6, 2, 2, 2, 2, 3, 2>("large-strided");
    packing();
    quantize();
    return 0;
}
