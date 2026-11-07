// tinymind's QSoftmaxFixedPoint1D and buildQSoftmaxFixedPointParams over a
// sweep of calibrations and row lengths. Prints the parameters and an FNV-1a
// hash of the outputs per calibration; tests/golden/softmax/stdout.txt is this
// program's output, and tests/softmax_sweep.rs computes the same in Rust.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "qsoftmax_fixedpoint.hpp"
#include "qcalibration.hpp"

static uint32_t gState;

static int8_t nextLogit(const std::size_t row)
{
    gState = gState * 1664525u + 1013904223u;
    const uint32_t v = gState >> 24;
    // Every fourth row in a narrow band, so logits tie or sit one step apart.
    return (row % 4 == 3) ? static_cast<int8_t>(96 + static_cast<int>(v & 7u))
                          : static_cast<int8_t>(static_cast<int>(v) - 128);
}

template<std::size_t N>
static uint32_t sweepRows(const int32_t multiplier, const int32_t leftShift, const int32_t diffMin, uint32_t hash)
{
    const tinymind::QSoftmaxFixedPoint1D<1, N> softmax = {multiplier, leftShift, diffMin};
    int8_t input[N];
    int8_t output[N];
    for (std::size_t r = 0; r < 32; ++r)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            input[i] = nextLogit(r);
        }
        softmax.forward(input, output);
        for (std::size_t i = 0; i < N; ++i)
        {
            hash ^= static_cast<uint8_t>(output[i]);
            hash *= 0x01000193u;
        }
    }
    return hash;
}

int main()
{
    const double betas[3] = {1.0, 0.5, 2.5};
    for (int e = -14; e <= 6; ++e)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int b = 0; b < 3; ++b)
            {
                // Exact in binary: (1 + j/8) * 2^e.
                const double scale = std::ldexp(1.0 + j / 8.0, e);
                int32_t multiplier = 0;
                int32_t leftShift = 0;
                int32_t diffMin = 0;
                tinymind::buildQSoftmaxFixedPointParams(scale, betas[b], multiplier, leftShift, diffMin);
                gState = static_cast<uint32_t>((e + 14) * 64 + j * 4 + b);
                uint32_t hash = 0x811C9DC5u;
                hash = sweepRows<1>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<2>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<3>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<10>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<12>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<64>(multiplier, leftShift, diffMin, hash);
                hash = sweepRows<511>(multiplier, leftShift, diffMin, hash);
                std::printf("e=%d j=%d beta=%d params=%d,%d,%d hash=0x%08X\n", e, j, b,
                            static_cast<int>(multiplier), static_cast<int>(leftShift),
                            static_cast<int>(diffMin), hash);
            }
        }
    }
    return 0;
}
