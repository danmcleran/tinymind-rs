// tinymind's inference-only NeuralNetwork (IsTrainable = false) with one to
// three hidden layers (HiddenLayers<...>), weights loaded through its setters,
// in five configurations: Q16.16 and Q8.8, tanh and ReLU hidden layers,
// sigmoid and linear outputs. The same weights also go into the trainable
// network, which must agree. tests/golden/inference_deep/stdout.txt is this
// program's output, and tests/inference_deep.rs computes the same with
// tinymind::InferenceNetwork.
//
// Per configuration: 8 weight sets, each drawn in layer order (input ->
// first hidden by source, then its bias; then each hidden layer -> the next by
// source, then its bias), raw values within +-2.0; then 16 inputs within
// +-2.0. The line hashes every output's raw value (FNV-1a over 4 bytes).
#include <cstddef>
#include <cstdint>
#include <cstdio>

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

template<typename ValueType>
static void absorb(uint32_t& hash, const ValueType& v)
{
    const uint32_t raw = static_cast<uint32_t>(static_cast<int32_t>(v.getValue()));
    for (int b = 0; b < 4; ++b)
    {
        hash ^= static_cast<uint8_t>(raw >> (8 * b));
        hash *= 0x01000193u;
    }
}

template<typename ValueType, template<typename> class Hidden, template<typename> class Output,
         size_t IN, typename HL, size_t L, size_t OUT>
static void run(const char* name, const size_t (&sizes)[L], const uint32_t seed)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, NoRandomWeights<ValueType>, Hidden<ValueType>,
                                                  Output<ValueType>, OUT> TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF, false> InferenceNet;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF, true> TrainableNet;
    const int32_t one = static_cast<int32_t>(tinymind::Constants<ValueType>::one().getValue());

    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    bool agree = true;
    for (int set = 0; set < 8; ++set)
    {
        static InferenceNet inference;
        static TrainableNet trainable;
        for (size_t i = 0; i < IN; ++i)
        {
            for (size_t h = 0; h < sizes[0]; ++h)
            {
                const ValueType w(static_cast<Raw>(nextRaw(one)));
                inference.setInputLayerWeightForNeuronAndConnection(i, h, w);
                trainable.setInputLayerWeightForNeuronAndConnection(i, h, w);
            }
        }
        for (size_t h = 0; h < sizes[0]; ++h)
        {
            const ValueType w(static_cast<Raw>(nextRaw(one)));
            inference.setInputLayerBiasWeightForConnection(h, w);
            trainable.setInputLayerBiasWeightForConnection(h, w);
        }
        for (size_t l = 0; l < L; ++l)
        {
            const size_t next = (l + 1 < L) ? sizes[l + 1] : OUT;
            for (size_t n = 0; n < sizes[l]; ++n)
            {
                for (size_t c = 0; c < next; ++c)
                {
                    const ValueType w(static_cast<Raw>(nextRaw(one)));
                    inference.setHiddenLayerWeightForNeuronAndConnection(l, n, c, w);
                    trainable.setHiddenLayerWeightForNeuronAndConnection(l, n, c, w);
                }
            }
            for (size_t c = 0; c < next; ++c)
            {
                const ValueType w(static_cast<Raw>(nextRaw(one)));
                inference.setHiddenLayerBiasNeuronWeightForConnection(l, c, w);
                trainable.setHiddenLayerBiasNeuronWeightForConnection(l, c, w);
            }
        }
        for (int s = 0; s < 16; ++s)
        {
            ValueType x[IN];
            for (size_t i = 0; i < IN; ++i)
            {
                x[i] = ValueType(static_cast<Raw>(nextRaw(one)));
            }
            ValueType y[OUT];
            ValueType z[OUT];
            inference.feedForward(x);
            inference.getLearnedValues(y);
            trainable.feedForward(x);
            trainable.getLearnedValues(z);
            for (size_t o = 0; o < OUT; ++o)
            {
                absorb(hash, y[o]);
                agree = agree && (y[o] == z[o]);
            }
        }
    }
    std::printf("%s hash=0x%08X trainable_agrees=%d\n", name, hash, agree ? 1 : 0);
}

int main()
{
    typedef tinymind::QValue<16, 16, true> Q16;
    typedef tinymind::QValue<8, 8, true> Q8;
    const size_t two4[] = {4, 4};
    const size_t three[] = {8, 6, 5};
    const size_t two86[] = {8, 6};
    const size_t one6[] = {6};
    run<Q16, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4, 4>, 2, 1>("q16.16 2-4-4-1 tanh/sigmoid", two4, 1);
    run<Q8, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4, 4>, 2, 1>("q8.8 2-4-4-1 tanh/sigmoid", two4, 2);
    run<Q16, tinymind::ReluActivationPolicy, tinymind::SigmoidActivationPolicy, 4,
        tinymind::HiddenLayers<8, 6, 5>, 3, 3>("q16.16 4-8-6-5-3 relu/sigmoid", three, 3);
    run<Q8, tinymind::ReluActivationPolicy, tinymind::SigmoidActivationPolicy, 4,
        tinymind::HiddenLayers<8, 6>, 2, 3>("q8.8 4-8-6-3 relu/sigmoid", two86, 4);
    run<Q16, tinymind::ReluActivationPolicy, tinymind::LinearActivationPolicy, 3,
        tinymind::HiddenLayers<6>, 1, 2>("q16.16 3-6-2 relu/linear", one6, 5);
    return 0;
}
