// tinymind's int8 composition layers and their calibration: QMul with
// buildQMulRequantizer, QConcat2_2D with buildRescaler, and QPad2D.
//
// QMul: 32 layers calibrated from random exact float scales, zero points and
// output ranges (full, fused ReLU, narrow, any), each run on all 65536 (a, b)
// pairs; then 48 layers with random integer requantizers on both of its paths
// (shift 2..31 and -3..1), run on random tensors. QConcat2_2D: five shapes,
// four trials each, rescalers calibrated from random scales. QPad2D: six
// shapes with padding on different sides, three trials each.
//
// Each line hashes every output byte (FNV-1a). The draws come in a fixed
// order, which tests/compose_sweep.rs repeats with the Rust layers.
// tests/golden/compose/stdout.txt is this program's output.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qaffine.hpp"
#include "qmul.hpp"
#include "qpad.hpp"
#include "qconcat.hpp"
#include "include/qcalibration.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

static uint32_t gState = 0x5EED0003u;

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

// (1 + d) * 2^-(16 + base + e), e in 0..3: exact in float, in (2^-(base + 3) * 2^-16, 2^-base].
static float randScale(const int base)
{
    const float m = static_cast<float>(1 + draw(1u << 16));
    return std::ldexp(m, -16 - base - static_cast<int>(draw(4)));
}

static uint32_t bits(const float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

static void randRange(const int8_t zero_point, int32_t& qmin, int32_t& qmax)
{
    const uint32_t kind = draw(4);
    if (kind == 0)
    {
        qmin = -128;
        qmax = 127;
    }
    else if (kind == 1)
    {
        qmin = zero_point; // fused ReLU
        qmax = 127;
    }
    else if (kind == 2)
    {
        qmin = (zero_point < -108) ? -128 : zero_point - 20;
        qmax = (zero_point > 107) ? 127 : zero_point + 20;
    }
    else
    {
        qmin = randI8(); // may exceed qmax: the result is then qmax
        qmax = randI8();
    }
}

static uint32_t fnv(uint32_t h, const int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        h ^= static_cast<uint8_t>(p[i]);
        h *= 0x01000193u;
    }
    return h;
}

static void printRq(const Rq& r)
{
    std::printf("rq=%d,%d,%d,%d,%d", r.multiplier, r.shift, r.zero_point, r.qmin, r.qmax);
}

static void fill(int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        p[i] = randI8();
    }
}

template<std::size_t H, std::size_t W, std::size_t CA, std::size_t CB>
static void concat(const char* name)
{
    typedef tinymind::QConcat2_2D<int8_t, int8_t, int8_t, H, W, CA, CB> Layer;
    for (int t = 0; t < 4; ++t)
    {
        Layer layer;
        layer.input_a_zero_point = randI8();
        layer.input_b_zero_point = randI8();
        const float sa = randScale(10);
        const float sb = randScale(10);
        const float so = randScale(10);
        const int8_t zo = randI8();
        int32_t qmin = 0, qmax = 0;
        randRange(zo, qmin, qmax);
        layer.requantizer_a = tinymind::buildRescaler<int8_t>(sa, so, zo, qmin, qmax);
        layer.requantizer_b = tinymind::buildRescaler<int8_t>(sb, so, zo, qmin, qmax);
        int8_t a[H * W * CA], b[H * W * CB], y[H * W * (CA + CB)];
        fill(a, H * W * CA);
        fill(b, H * W * CB);
        layer.forward(a, b, y);
        std::printf("concat %s %d scales=0x%08X,0x%08X,0x%08X zp=%d,%d ", name, t, bits(sa), bits(sb),
                    bits(so), layer.input_a_zero_point, layer.input_b_zero_point);
        printRq(layer.requantizer_a);
        std::printf(" ");
        printRq(layer.requantizer_b);
        std::printf(" hash=0x%08X\n", fnv(0x811C9DC5u, y, H * W * (CA + CB)));
    }
}

template<std::size_t H, std::size_t W, std::size_t C, std::size_t PT, std::size_t PB, std::size_t PL,
         std::size_t PR>
static void pad(const char* name)
{
    typedef tinymind::QPad2D<int8_t, H, W, C, PT, PB, PL, PR> Layer;
    for (int t = 0; t < 3; ++t)
    {
        Layer layer;
        layer.pad_value = randI8();
        int8_t x[Layer::InputSize], y[Layer::OutputSize];
        fill(x, Layer::InputSize);
        layer.forward(x, y);
        std::printf("pad %s %d value=%d hash=0x%08X\n", name, t, layer.pad_value,
                    fnv(0x811C9DC5u, y, Layer::OutputSize));
    }
}

int main()
{
    typedef tinymind::QMul<int8_t, int8_t, int8_t, 256> Mul;
    for (int k = 0; k < 32; ++k)
    {
        // Scales around 2^-4..2^-7 each, the output around 2^-6..2^-9: products
        // spread over the int8 range rather than all saturating.
        const float sa = randScale(4);
        const float sb = randScale(4);
        const float so = randScale(6);
        Mul mul;
        mul.input_a_zero_point = randI8();
        mul.input_b_zero_point = randI8();
        const int8_t zo = randI8();
        int32_t qmin = 0, qmax = 0;
        randRange(zo, qmin, qmax);
        mul.requantizer = tinymind::buildQMulRequantizer<int8_t>(sa, sb, so, zo, qmin, qmax);
        uint32_t h = 0x811C9DC5u;
        int8_t a[256], b[256], y[256];
        for (int j = 0; j < 256; ++j)
        {
            b[j] = static_cast<int8_t>(j - 128);
        }
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 256; ++j)
            {
                a[j] = static_cast<int8_t>(i - 128);
            }
            mul.forward(a, b, y);
            h = fnv(h, y, 256);
        }
        std::printf("mul calibrated %d scales=0x%08X,0x%08X,0x%08X zp=%d,%d ", k, bits(sa), bits(sb), bits(so),
                    mul.input_a_zero_point, mul.input_b_zero_point);
        printRq(mul.requantizer);
        std::printf(" hash=0x%08X\n", h);
    }
    for (int k = 0; k < 48; ++k)
    {
        Mul mul;
        mul.input_a_zero_point = randI8();
        mul.input_b_zero_point = randI8();
        mul.requantizer.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        // Products reach 2^16: shifts 8..15 spread them over the output range;
        // a third take -3..1, the two-step path, where most saturate.
        mul.requantizer.shift = (draw(3) == 0) ? static_cast<int32_t>(draw(5)) - 3
                                               : 8 + static_cast<int32_t>(draw(8));
        mul.requantizer.zero_point = randI8();
        int32_t qmin = 0, qmax = 0;
        randRange(mul.requantizer.zero_point, qmin, qmax);
        mul.requantizer.qmin = static_cast<int8_t>(qmin);
        mul.requantizer.qmax = static_cast<int8_t>(qmax);
        int8_t a[256], b[256], y[256];
        fill(a, 256);
        fill(b, 256);
        mul.forward(a, b, y);
        std::printf("mul direct %d zp=%d,%d ", k, mul.input_a_zero_point, mul.input_b_zero_point);
        printRq(mul.requantizer);
        std::printf(" hash=0x%08X\n", fnv(0x811C9DC5u, y, 256));
    }

    concat<3, 4, 2, 3>("3x4_2+3");
    concat<1, 1, 1, 1>("1x1_1+1");
    concat<5, 2, 4, 1>("5x2_4+1");
    concat<2, 3, 8, 8>("2x3_8+8");
    concat<4, 4, 16, 3>("4x4_16+3");

    pad<3, 4, 2, 1, 1, 1, 1>("3x4x2_1111");
    pad<2, 2, 3, 0, 2, 3, 0>("2x2x3_0230");
    pad<4, 1, 1, 2, 0, 0, 1>("4x1x1_2001");
    pad<1, 5, 4, 1, 1, 0, 0>("1x5x4_1100");
    pad<2, 3, 8, 0, 0, 0, 0>("2x3x8_none");
    pad<3, 3, 1, 4, 4, 4, 4>("3x3x1_4444");
    return 0;
}
