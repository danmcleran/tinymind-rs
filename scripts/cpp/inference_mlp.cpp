// tinymind's inference-only MultilayerPerceptron (IsTrainable = false) with
// weights loaded through its setters, over four configurations (Q16.16 and
// Q8.8; the xor shape 2-4-1 tanh/sigmoid and the iris shape 4-8-3
// relu/sigmoid). The same weights also go into the trainable network, which
// must agree. tests/golden/inference_mlp/stdout.txt is this program's output,
// and tests/inference_mlp.rs computes the same with tinymind::InferencePerceptron.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"

static uint32_t gState;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

// A raw value within +-2.0 for a format whose 1.0 is `one` raw units.
static int32_t nextRaw(const int32_t one)
{
    const uint32_t span = static_cast<uint32_t>(4 * one + 1);
    return static_cast<int32_t>((nextU32() >> 8) % span) - 2 * one;
}

template<typename ValueType>
struct NoRandomWeights
{
    static ValueType generateRandomWeight()
    {
        return ValueType(0);
    }
};

template<typename ValueType, template<typename> class Hidden, size_t IN, size_t HID, size_t OUT>
static void run(const char* name, const int32_t one)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, NoRandomWeights<ValueType>, Hidden<ValueType>,
                                                  tinymind::SigmoidActivationPolicy<ValueType>, OUT> TF;
    typedef tinymind::MultilayerPerceptron<ValueType, IN, 1, HID, OUT, TF, false> InferenceNet;
    typedef tinymind::MultilayerPerceptron<ValueType, IN, 1, HID, OUT, TF, true> TrainableNet;

    uint32_t hash = 0x811C9DC5u;
    bool agree = true;
    for (int set = 0; set < 8; ++set)
    {
        static InferenceNet inference;
        static TrainableNet trainable;
        for (size_t i = 0; i < IN; ++i)
        {
            for (size_t h = 0; h < HID; ++h)
            {
                const ValueType w(static_cast<Raw>(nextRaw(one)));
                inference.setInputLayerWeightForNeuronAndConnection(i, h, w);
                trainable.setInputLayerWeightForNeuronAndConnection(i, h, w);
            }
        }
        for (size_t h = 0; h < HID; ++h)
        {
            const ValueType w(static_cast<Raw>(nextRaw(one)));
            inference.setInputLayerBiasWeightForConnection(h, w);
            trainable.setInputLayerBiasWeightForConnection(h, w);
        }
        for (size_t h = 0; h < HID; ++h)
        {
            for (size_t o = 0; o < OUT; ++o)
            {
                const ValueType w(static_cast<Raw>(nextRaw(one)));
                inference.setHiddenLayerWeightForNeuronAndConnection(0, h, o, w);
                trainable.setHiddenLayerWeightForNeuronAndConnection(0, h, o, w);
            }
        }
        for (size_t o = 0; o < OUT; ++o)
        {
            const ValueType w(static_cast<Raw>(nextRaw(one)));
            inference.setHiddenLayerBiasNeuronWeightForConnection(0, o, w);
            trainable.setHiddenLayerBiasNeuronWeightForConnection(0, o, w);
        }

        for (int sample = 0; sample < 256; ++sample)
        {
            ValueType input[IN];
            for (size_t i = 0; i < IN; ++i)
            {
                input[i] = ValueType(static_cast<Raw>(nextRaw(one)));
            }
            ValueType out[OUT];
            ValueType check[OUT];
            inference.feedForward(input);
            inference.getLearnedValues(out);
            trainable.feedForward(input);
            trainable.getLearnedValues(check);
            for (size_t o = 0; o < OUT; ++o)
            {
                const int32_t raw = static_cast<int32_t>(out[o].getValue());
                agree = agree && (raw == static_cast<int32_t>(check[o].getValue()));
                for (int b = 0; b < 4; ++b)
                {
                    hash ^= static_cast<uint8_t>(static_cast<uint32_t>(raw) >> (8 * b));
                    hash *= 0x01000193u;
                }
                if (set == 0 && sample < 2)
                {
                    std::printf("%s set=0 sample=%d out%d=%d\n", name, sample, static_cast<int>(o),
                                static_cast<int>(raw));
                }
            }
        }
    }
    std::printf("%s hash=0x%08X trainable_agrees=%s\n", name, hash, agree ? "yes" : "no");
}

int main()
{
    typedef tinymind::QValue<16, 16, true> Q16;
    typedef tinymind::QValue<8, 8, true> Q8;
    gState = 1;
    run<Q16, tinymind::TanhActivationPolicy, 2, 4, 1>("q16.16 2-4-1 tanh/sigmoid", 1 << 16);
    gState = 2;
    run<Q8, tinymind::TanhActivationPolicy, 2, 4, 1>("q8.8 2-4-1 tanh/sigmoid", 1 << 8);
    gState = 3;
    run<Q16, tinymind::ReluActivationPolicy, 4, 8, 3>("q16.16 4-8-3 relu/sigmoid", 1 << 16);
    gState = 4;
    run<Q8, tinymind::ReluActivationPolicy, 4, 8, 3>("q8.8 4-8-3 relu/sigmoid", 1 << 8);
    return 0;
}
