// tinymind's LstmNeuralNetwork (one LSTM hidden layer) trained on short
// sequences, in Q16.16 and Q8.8, with tanh or ReLU cell candidates, sigmoid or
// linear outputs, gradient clipping and the Adam optimizer. Weights come from
// glibc rand() as tinymind's examples draw them; inputs from an LCG.
//
// Each run feeds sequences of 1 to 12 steps, calling resetState() before each
// one. After every step it computes the error, trains unless the error is
// within zero tolerance or the step is a multiple of 7 (so some steps only
// feed forward and the recurrent state carries on untrained), and an FNV-1a
// hash absorbs the outputs and every weight: input -> gates (by input, then
// connection h*4+gate), the input bias, recurrent -> gates, hidden -> output
// and the hidden bias. tests/golden/lstm/stdout.txt is this program's output;
// tests/lstm_sweep.rs runs the same sequences with tinymind::LstmNetwork.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"
#include "gradientClipping.hpp"
#include "adam.hpp"

static uint32_t gState;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

template<typename ValueType>
struct QuarterWeights
{
    // har_activity's generator: symmetric in [-0.25, 0.25], computed in int.
    static ValueType generateRandomWeight()
    {
        typedef typename ValueType::FullWidthValueType Raw;
        const int quarter = static_cast<int>(tinymind::Constants<ValueType>::one().getValue()) / 4;
        return ValueType(static_cast<Raw>((rand() % (2 * quarter + 1)) - quarter));
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

template<typename NN, typename ValueType, size_t IN, size_t HID, size_t OUT>
static void absorbWeights(NN& nn, uint32_t& hash)
{
    for (size_t i = 0; i < IN; ++i)
        for (size_t c = 0; c < 4 * HID; ++c)
            absorb(hash, nn.getInputLayerWeightForNeuronAndConnection(i, c));
    for (size_t c = 0; c < 4 * HID; ++c)
        absorb(hash, nn.getInputLayerBiasNeuronWeightForConnection(c));
    for (size_t j = 0; j < HID; ++j)
        for (size_t c = 0; c < 4 * HID; ++c)
            absorb(hash, nn.getRecurrentLayer().getWeightForNeuronAndConnection(j, c));
    for (size_t j = 0; j < HID; ++j)
        for (size_t o = 0; o < OUT; ++o)
            absorb(hash, nn.getHiddenLayerWeightForNeuronAndConnection(0, j, o));
    for (size_t o = 0; o < OUT; ++o)
        absorb(hash, nn.getHiddenLayerBiasNeuronWeightForConnection(0, o));
}

// Inputs in [-1, 1). Targets need the previous step: output 0 is 1 when the
// previous step's first input was positive, output 1 when the sign of the
// first input changed, output k >= 2 when input k % IN is above the previous
// first input. With a linear output the targets are scaled by `targetScale`.
template<typename ValueType, size_t IN, size_t OUT>
static void sample(ValueType (&x)[IN], ValueType (&t)[OUT], int& previous, const int one, const int targetScale)
{
    typedef typename ValueType::FullWidthValueType Raw;
    int v[IN];
    for (size_t i = 0; i < IN; ++i)
    {
        v[i] = static_cast<int>((nextU32() >> 8) % static_cast<uint32_t>(2 * one)) - one;
        x[i] = ValueType(static_cast<Raw>(v[i]));
    }
    for (size_t o = 0; o < OUT; ++o)
    {
        int bit;
        if (o == 0)
            bit = previous > 0;
        else if (o == 1)
            bit = (previous > 0) != (v[0] > 0);
        else
            bit = v[o % IN] > previous;
        t[o] = ValueType(static_cast<Raw>(bit * one * targetScale));
    }
    previous = v[0];
}

template<typename ValueType, template<typename> class Hidden, template<typename> class Output, size_t IN, size_t HID,
         size_t OUT, typename Clip, typename Opt = tinymind::NullOptimizerPolicy<ValueType> >
static void run(const char* name, const unsigned seed, const int steps, const int lrRaw, const int targetScale)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, QuarterWeights<ValueType>, Hidden<ValueType>,
                                                  Output<ValueType>, OUT,
                                                  tinymind::DefaultNetworkInitializer<ValueType>,
                                                  tinymind::MeanSquaredErrorCalculator<ValueType, OUT>,
                                                  tinymind::ZeroToleranceCalculator<ValueType>, Clip,
                                                  tinymind::NullWeightDecayPolicy<ValueType>,
                                                  tinymind::FixedLearningRatePolicy<ValueType>, Opt> TF;
    typedef tinymind::LstmNeuralNetwork<ValueType, IN, tinymind::HiddenLayers<HID>, OUT, TF> NN;
    const int one = static_cast<int>(tinymind::Constants<ValueType>::one().getValue());

    srand(seed);
    static NN nn;
    if (lrRaw != 0)
    {
        nn.setLearningRate(ValueType(static_cast<Raw>(lrRaw)));
    }
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    ValueType y[OUT];
    int step = 0;
    int trained = 0;
    while (step < steps)
    {
        const int length = 1 + static_cast<int>((nextU32() >> 8) % 12u);
        int previous = 0;
        nn.resetState();
        for (int k = 0; k < length && step < steps; ++k)
        {
            ++step;
            sample<ValueType, IN, OUT>(x, t, previous, one, targetScale);
            nn.feedForward(x);
            const ValueType error = nn.calculateError(t);
            if (step % 7 != 0 && !TF::isWithinZeroTolerance(error))
            {
                nn.trainNetwork(t);
                ++trained;
            }
            nn.getLearnedValues(y);
            for (size_t o = 0; o < OUT; ++o)
                absorb(hash, y[o]);
            absorbWeights<NN, ValueType, IN, HID, OUT>(nn, hash);
            if (step % 250 == 0)
            {
                std::printf("%s step=%d trained=%d error=%d hash=0x%08X\n", name, step, trained,
                            static_cast<int>(error.getValue()), hash);
            }
        }
    }
}

int main()
{
    typedef tinymind::QValue<16, 16, true> Q16;
    typedef tinymind::QValue<8, 8, true> Q8;
    typedef tinymind::AdamOptimizer<Q16, 0, 58982, 0, 65470, 0, 16> Adam16;

    // har_activity's network: 3 inputs, 16 LSTM cells, 4 sigmoid outputs, clipped.
    run<Q16, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 3, 16, 4,
        tinymind::GradientClipByValue<Q16> >("q16.16 3-16-4 tanh/sigmoid clip[-1,1]", 1, 1500, 0, 1);
    run<Q16, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2, 6, 3,
        tinymind::NullGradientClippingPolicy<Q16> >("q16.16 2-6-3 tanh/sigmoid", 2, 2000, 0, 1);
    run<Q16, tinymind::ReluActivationPolicy, tinymind::LinearActivationPolicy, 2, 5, 2,
        tinymind::GradientClipByValue<Q16> >("q16.16 2-5-2 relu/linear clip[-1,1]", 3, 2000, 1 << 10, 3);
    run<Q8, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2, 4, 2,
        tinymind::GradientClipByValue<Q8> >("q8.8 2-4-2 tanh/sigmoid clip[-1,1]", 4, 2000, 0, 1);
    run<Q8, tinymind::ReluActivationPolicy, tinymind::SigmoidActivationPolicy, 3, 6, 3,
        tinymind::GradientClipByValue<Q8, 2, 0x80> >("q8.8 3-6-3 relu/sigmoid clip[-1.5,2.5]", 5, 2000, 0, 1);
    run<Q16, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 1, 1, 1,
        tinymind::NullGradientClippingPolicy<Q16> >("q16.16 1-1-1 tanh/sigmoid", 6, 1000, 0, 1);
    run<Q16, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2, 4, 2,
        tinymind::NullGradientClippingPolicy<Q16>, Adam16>("q16.16 2-4-2 tanh/sigmoid adam", 7, 2000, 1 << 11, 1);
    return 0;
}
