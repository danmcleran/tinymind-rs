// The importer example's model (tools/import_pytorch/example_mlp.py) on
// tinymind's C++ int8 runtime, from the weights.hpp tinymind's own importer
// writes. It builds the requantizers and the sigmoid table at startup as
// tinymind's docs describe, then prints what examples/imported_mlp.rs prints,
// so scripts/compare-with-cpp.sh can byte-compare the two.
#include <cstdint>
#include <cstdio>

#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"
#include "qdense.hpp"

#include "weights.hpp"

namespace m = example_mlp;

template<typename Layer>
static void printRq(const char* name, const Layer& layer)
{
    std::printf("%s rq=%d,%d,%d,%d,%d\n", name, static_cast<int>(layer.requantizer.multiplier),
                static_cast<int>(layer.requantizer.shift), static_cast<int>(layer.requantizer.zero_point),
                static_cast<int>(layer.requantizer.qmin), static_cast<int>(layer.requantizer.qmax));
}

int main()
{
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 3, 8> fc1;
    fc1.weights = m::kfc1_Weights;
    fc1.biases = m::kfc1_Biases;
    fc1.input_zero_point = static_cast<int8_t>(m::kinput_ZeroPoint);
    fc1.requantizer = tinymind::buildRequantizer<int8_t>(m::kinput_Scale, m::kfc1_WeightScale, m::kh1_Scale, m::kh1_ZeroPoint, -128, 127);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 8, 4> fc2;
    fc2.weights = m::kfc2_Weights;
    fc2.biases = m::kfc2_Biases;
    fc2.input_zero_point = static_cast<int8_t>(m::kh1_ZeroPoint);
    fc2.requantizer = tinymind::buildRequantizer<int8_t>(m::kh1_Scale, m::kfc2_WeightScale, m::kh2_Scale, m::kh2_ZeroPoint, -128, 127);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 4, 2> fc3;
    fc3.weights = m::kfc3_Weights;
    fc3.biases = m::kfc3_Biases;
    fc3.input_zero_point = static_cast<int8_t>(m::kh2_ZeroPoint);
    fc3.requantizer = tinymind::buildRequantizer<int8_t>(m::kh2_Scale, m::kfc3_WeightScale, m::kfc3_Scale, m::kfc3_ZeroPoint, -128, 127);

    int8_t lut[tinymind::kQActivationLUTSize];
    tinymind::buildQSigmoidLUT(m::kfc3_Scale, m::kfc3_ZeroPoint, m::kout_Scale, m::kout_ZeroPoint, lut);

    printRq("fc1", fc1);
    printRq("fc2", fc2);
    printRq("fc3", fc3);
    uint32_t h = 0x811C9DC5u;
    for (const int8_t v : lut)
    {
        h ^= static_cast<uint8_t>(v);
        h *= 0x01000193u;
    }
    std::printf("lut fnv=0x%08X\n", h);

    for (int s = 0; s < 16; ++s)
    {
        int8_t in[3];
        for (int k = 0; k < 3; ++k)
        {
            const float x = static_cast<float>((s * 23 + k * 29) % 37) / 18.0f - 1.0f;
            in[k] = tinymind::quantize<int8_t>(x, m::kinput_Scale, m::kinput_ZeroPoint, -128, 127);
        }
        int8_t h1[8];
        int8_t h2[4];
        int8_t lg[2];
        fc1.forward(in, h1);
        tinymind::qreluBuffer(h1, 8, static_cast<int8_t>(m::kh1_ZeroPoint));
        fc2.forward(h1, h2);
        tinymind::qreluBuffer(h2, 4, static_cast<int8_t>(m::kh2_ZeroPoint));
        fc3.forward(h2, lg);
        std::printf("in=%d,%d,%d out=%d,%d\n", in[0], in[1], in[2],
                    tinymind::qApplyLUT(lg[0], lut), tinymind::qApplyLUT(lg[1], lut));
    }
    return 0;
}
