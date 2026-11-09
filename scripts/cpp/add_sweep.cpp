// tinymind's int8 elementwise addition, QAdd, and its calibration,
// buildQAddParams.
//
// Two kinds of case. Calibrated: random input and output scales (exact
// floats, so the Rust draws the same ones), zero points and output ranges
// (full, fused ReLU, narrow, or any, so sometimes qmin > qmax);
// buildQAddParams turns the scales into the integer parameters, which are
// printed with the scales' bits, and the layer runs on every one of the 65536
// (a, b) pairs. Direct: random multipliers, shifts and left
// shifts written straight into the layer, kept where the C++ is defined (the
// rescaled inputs and their sum stay inside int32), with output shifts on
// both of the requantizer's paths, run on random tensors. Saturating: sums near
// the int32 rails with an output left shift, which saturates.
//
// Each line hashes every output byte (FNV-1a). The draws come in a fixed
// order, which tests/add_sweep.rs repeats with the Rust layer.
// tests/golden/add/stdout.txt is this program's output.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qaffine.hpp"
#include "qadd.hpp"
#include "include/qcalibration.hpp"

typedef tinymind::QAdd<int8_t, int8_t, int8_t, 256> Add;

static uint32_t gState = 0x5EED0002u;

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

// (1 + d) * 2^-(16 + e): exact in float, in (2^-19, 1].
static float randScale()
{
    const float m = static_cast<float>(1 + draw(1u << 16));
    return std::ldexp(m, -16 - static_cast<int>(draw(4)));
}

static uint32_t bits(const float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

static void randRange(const int8_t zero_point, int8_t& qmin, int8_t& qmax)
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
        // A narrow range around the zero point.
        qmin = static_cast<int8_t>((zero_point < -108) ? -128 : zero_point - 20);
        qmax = static_cast<int8_t>((zero_point > 107) ? 127 : zero_point + 20);
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

static void print(const char* kind, const int k, const Add& add, const uint32_t h)
{
    std::printf("%s %d zp=%d,%d ls=%d a=%d,%d b=%d,%d rq=%d,%d,%d,%d,%d hash=0x%08X\n", kind, k,
                add.input_a_zero_point, add.input_b_zero_point, add.left_shift,
                add.input_a_multiplier, add.input_a_shift, add.input_b_multiplier,
                add.input_b_shift, add.output_requantizer.multiplier,
                add.output_requantizer.shift, add.output_requantizer.zero_point,
                add.output_requantizer.qmin, add.output_requantizer.qmax, h);
}

int main()
{
    for (int k = 0; k < 48; ++k)
    {
        const float sa = randScale();
        const float sb = (draw(4) == 0) ? sa : randScale(); // equal scales a quarter of the time
        const float so = randScale();
        Add add;
        add.input_a_zero_point = randI8();
        add.input_b_zero_point = randI8();
        const int8_t zo = randI8();
        int8_t qmin = 0, qmax = 0;
        randRange(zo, qmin, qmax);
        const tinymind::QAddParams p = tinymind::buildQAddParams(sa, sb, so);
        add.left_shift = p.left_shift;
        add.input_a_multiplier = p.input_a_multiplier;
        add.input_a_shift = p.input_a_shift;
        add.input_b_multiplier = p.input_b_multiplier;
        add.input_b_shift = p.input_b_shift;
        add.output_requantizer.multiplier = p.output_multiplier;
        add.output_requantizer.shift = p.output_shift;
        add.output_requantizer.zero_point = zo;
        add.output_requantizer.qmin = qmin;
        add.output_requantizer.qmax = qmax;

        // Every (a, b) pair: a row per value of a, b running over all 256.
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
            add.forward(a, b, y);
            h = fnv(h, y, 256);
        }
        std::printf("scales 0x%08X 0x%08X 0x%08X\n", bits(sa), bits(sb), bits(so));
        print("calibrated", k, add, h);
    }

    for (int k = 0; k < 64; ++k)
    {
        Add add;
        add.input_a_zero_point = randI8();
        add.input_b_zero_point = randI8();
        // |a - zp| < 2^8, so a left shift of at most 21 and an input left shift
        // of at most 1 keep each rescaled input below 2^30 and the sum in int32.
        add.left_shift = static_cast<int32_t>(draw(22));
        add.input_a_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        add.input_a_shift = static_cast<int32_t>(draw(12)) - 1;
        add.input_b_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        add.input_b_shift = static_cast<int32_t>(draw(12)) - 1;
        add.output_requantizer.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        // Each rescaled input is about (a - zp) * 2^(left_shift - shift): an
        // output shift near left_shift - min(shifts) spreads the outputs over
        // the range. A third of the cases take a shift of -3..1 instead, the
        // requantizer's two-step path, where most outputs saturate.
        const int32_t min_shift = (add.input_a_shift < add.input_b_shift)
            ? add.input_a_shift : add.input_b_shift;
        int32_t out_shift = add.left_shift - min_shift + static_cast<int32_t>(draw(5)) - 2;
        if (draw(3) == 0 || out_shift < -3)
        {
            out_shift = static_cast<int32_t>(draw(5)) - 3;
        }
        add.output_requantizer.shift = (out_shift > 31) ? 31 : out_shift;
        add.output_requantizer.zero_point = randI8();
        randRange(add.output_requantizer.zero_point, add.output_requantizer.qmin,
                  add.output_requantizer.qmax);

        int8_t a[256], b[256], y[256];
        for (int i = 0; i < 256; ++i)
        {
            a[i] = randI8();
            b[i] = randI8();
        }
        add.forward(a, b, y);
        print("direct", k, add, fnv(0x811C9DC5u, y, 256));
    }

    // The sum near the int32 rails, then an output left shift of 1 to 3: the
    // requantizer's two-step path saturates the shift for the larger sums.
    for (int k = 0; k < 16; ++k)
    {
        Add add;
        add.input_a_zero_point = randI8();
        add.input_b_zero_point = randI8();
        add.left_shift = 18 + static_cast<int32_t>(draw(4)); // 18..21
        add.input_a_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        add.input_a_shift = static_cast<int32_t>(draw(2)) - 1; // -1..0
        add.input_b_multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        add.input_b_shift = static_cast<int32_t>(draw(2)) - 1;
        add.output_requantizer.multiplier = static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
        add.output_requantizer.shift = -1 - static_cast<int32_t>(draw(3)); // -3..-1
        add.output_requantizer.zero_point = randI8();
        add.output_requantizer.qmin = -128;
        add.output_requantizer.qmax = 127;

        int8_t a[256], b[256], y[256];
        for (int i = 0; i < 256; ++i)
        {
            a[i] = randI8();
            b[i] = randI8();
        }
        add.forward(a, b, y);
        print("saturating", k, add, fnv(0x811C9DC5u, y, 256));
    }
    return 0;
}
