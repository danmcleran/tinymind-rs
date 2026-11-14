// tinymind's NeuralNetwork with one, two and three hidden layers (HiddenLayers<...>),
// trained with its default SGD with momentum, plus two regressions with a linear
// output and gradient clipping, runs with the Adam and RMSprop optimizers, and
// runs with L2 weight decay, a step-decay learning-rate schedule and early
// stopping. Weights are initialised from
// glibc rand() as tinymind's examples do; samples come from an LCG. After every
// training step an FNV-1a hash absorbs every weight, so any difference at any
// step changes the printed hashes. tests/golden/deep_mlp/stdout.txt is this
// program's output, and tests/deep_mlp.rs runs the same training with
// tinymind::NeuralNetwork.
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
#include "rmsprop.hpp"
#include "weightDecay.hpp"
#include "learningRateSchedule.hpp"
#include "earlyStopping.hpp"

static uint32_t gState;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

template<typename ValueType>
struct RandomWeights
{
    // tinymind's examples: symmetric in [-0.5, 0.5], computed in int, then narrowed.
    static ValueType generateRandomWeight()
    {
        typedef typename ValueType::FullWidthValueType Raw;
        const int half = static_cast<int>(tinymind::Constants<ValueType>::one().getValue()) / 2;
        return ValueType(static_cast<Raw>((rand() % (2 * half + 1)) - half));
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

// Every weight, in layer order: input -> first hidden (by source, then the
// bias), then each hidden layer -> the next layer.
template<typename NN, typename ValueType, size_t IN, size_t L>
static void absorbWeights(NN& nn, uint32_t& hash, const size_t (&sizes)[L], const size_t out)
{
    for (size_t i = 0; i < IN; ++i)
        for (size_t h = 0; h < sizes[0]; ++h)
            absorb(hash, nn.getInputLayerWeightForNeuronAndConnection(i, h));
    for (size_t h = 0; h < sizes[0]; ++h)
        absorb(hash, nn.getInputLayerBiasNeuronWeightForConnection(h));
    for (size_t l = 0; l < L; ++l)
    {
        const size_t next = (l + 1 < L) ? sizes[l + 1] : out;
        for (size_t n = 0; n < sizes[l]; ++n)
            for (size_t c = 0; c < next; ++c)
                absorb(hash, nn.getHiddenLayerWeightForNeuronAndConnection(l, n, c));
        for (size_t c = 0; c < next; ++c)
            absorb(hash, nn.getHiddenLayerBiasNeuronWeightForConnection(l, c));
    }
}

// A sample: XOR of two bits for 2 inputs; for 4 inputs, inputs in [0, 1) and a
// one-hot target for the largest of three pair sums.
template<typename ValueType, size_t IN, size_t OUT>
static void sample(ValueType (&x)[IN], ValueType (&t)[OUT], const int one)
{
    typedef typename ValueType::FullWidthValueType Raw;
    if (IN == 2)
    {
        const int a = static_cast<int>((nextU32() >> 16) & 1u);
        const int b = static_cast<int>((nextU32() >> 16) & 1u);
        x[0] = ValueType(static_cast<Raw>(a * one));
        x[1] = ValueType(static_cast<Raw>(b * one));
        t[0] = ValueType(static_cast<Raw>((a ^ b) * one));
        return;
    }
    int v[IN];
    for (size_t i = 0; i < IN; ++i)
    {
        v[i] = static_cast<int>((nextU32() >> 8) % static_cast<uint32_t>(one));
        x[i] = ValueType(static_cast<Raw>(v[i]));
    }
    const int s[3] = {v[0] + v[1], v[2] + v[3], v[0] + v[3]};
    size_t best = 0;
    for (size_t k = 1; k < 3; ++k)
        if (s[k] > s[best])
            best = k;
    for (size_t o = 0; o < OUT; ++o)
        t[o] = ValueType(static_cast<Raw>(o == best ? one : 0));
}

template<typename ValueType, template<typename> class Hidden, size_t IN, typename HL, size_t L, size_t OUT>
static void run(const char* name, const size_t (&sizes)[L], const unsigned seed, const int steps)
{
    typedef tinymind::FixedPointTransferFunctions<ValueType, RandomWeights<ValueType>, Hidden<ValueType>,
                                                  tinymind::SigmoidActivationPolicy<ValueType>, OUT> TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF> NN;
    const int one = static_cast<int>(tinymind::Constants<ValueType>::one().getValue());

    srand(seed);
    static NN nn;
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    for (int step = 1; step <= steps; ++step)
    {
        sample<ValueType, IN, OUT>(x, t, one);
        nn.feedForward(x);
        const ValueType error = nn.calculateError(t);
        nn.trainNetwork(t);
        absorbWeights<NN, ValueType, IN, L>(nn, hash, sizes, OUT);
        if (step % 500 == 0)
        {
            std::printf("%s step=%d error=%d hash=0x%08X\n", name, step, static_cast<int>(error.getValue()), hash);
        }
    }
    ValueType y[OUT];
    for (int k = 0; k < 4; ++k)
    {
        sample<ValueType, IN, OUT>(x, t, one);
        nn.feedForward(x);
        nn.getLearnedValues(y);
        std::printf("%s final%d", name, k);
        for (size_t o = 0; o < OUT; ++o)
            std::printf(" %d", static_cast<int>(y[o].getValue()));
        std::printf("\n");
    }
}

// An asymmetric clip range [MinRaw, MaxRaw] (raw values), tested in
// GradientClipByValue's order. Since tinymind #264 GradientClipByValue is
// symmetric ([-b, b]); the Rust API takes any (min, max), so the asymmetric
// case is kept with this policy.
template<typename ValueType, long long MinRaw, long long MaxRaw>
struct AsymmetricClip
{
    static ValueType clip(const ValueType& gradient)
    {
        static const ValueType maxValue(static_cast<typename ValueType::FullWidthValueType>(MaxRaw));
        static const ValueType minValue(static_cast<typename ValueType::FullWidthValueType>(MinRaw));
        if (gradient > maxValue)
        {
            return maxValue;
        }
        else if (gradient < minValue)
        {
            return minValue;
        }
        return gradient;
    }
};

// Regression with a linear output and gradient clipping: 3 inputs in [-1, 1),
// two targets up to about 9 in magnitude, so the output and bias gradients go
// well past the clip range (tinymind clips weight and bias gradients alike
// since its #264).
template<typename ValueType, size_t IN, typename HL, size_t L, size_t OUT, typename Clip,
         typename Opt = tinymind::NullOptimizerPolicy<ValueType> >
static void runRegression(const char* name, const size_t (&sizes)[L], const unsigned seed, const int steps)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, RandomWeights<ValueType>,
                                                  tinymind::ReluActivationPolicy<ValueType>,
                                                  tinymind::LinearActivationPolicy<ValueType>, OUT,
                                                  tinymind::DefaultNetworkInitializer<ValueType>,
                                                  tinymind::MeanSquaredErrorCalculator<ValueType, OUT>,
                                                  tinymind::ZeroToleranceCalculator<ValueType>, Clip,
                                                  tinymind::NullWeightDecayPolicy<ValueType>,
                                                  tinymind::FixedLearningRatePolicy<ValueType>, Opt> TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF> NN;
    const int one = static_cast<int>(tinymind::Constants<ValueType>::one().getValue());

    srand(seed);
    static NN nn;
    nn.setLearningRate(ValueType(static_cast<Raw>(one / 64)));
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    for (int step = 1; step <= steps; ++step)
    {
        int v[IN];
        for (size_t i = 0; i < IN; ++i)
        {
            v[i] = static_cast<int>((nextU32() >> 8) % static_cast<uint32_t>(2 * one)) - one;
            x[i] = ValueType(static_cast<Raw>(v[i]));
        }
        t[0] = ValueType(static_cast<Raw>(3 * v[0] - 2 * v[1] + 4 * v[2]));
        t[1] = ValueType(static_cast<Raw>(-2 * v[0] + 3 * v[2] + 2 * one));
        nn.feedForward(x);
        const ValueType error = nn.calculateError(t);
        nn.trainNetwork(t);
        absorbWeights<NN, ValueType, IN, L>(nn, hash, sizes, OUT);
        if (step % 500 == 0)
        {
            std::printf("%s step=%d error=%d hash=0x%08X\n", name, step, static_cast<int>(error.getValue()), hash);
        }
    }
}

// run() with an adaptive optimizer (Adam or RMSprop) and an explicit learning
// rate in raw units.
template<typename ValueType, template<typename> class Hidden, size_t IN, typename HL, size_t L, size_t OUT, typename Opt>
static void runAdaptive(const char* name, const size_t (&sizes)[L], const unsigned seed, const int steps, const int lrRaw)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, RandomWeights<ValueType>, Hidden<ValueType>,
                                                  tinymind::SigmoidActivationPolicy<ValueType>, OUT,
                                                  tinymind::DefaultNetworkInitializer<ValueType>,
                                                  tinymind::MeanSquaredErrorCalculator<ValueType, OUT>,
                                                  tinymind::ZeroToleranceCalculator<ValueType>,
                                                  tinymind::NullGradientClippingPolicy<ValueType>,
                                                  tinymind::NullWeightDecayPolicy<ValueType>,
                                                  tinymind::FixedLearningRatePolicy<ValueType>, Opt> TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF> NN;
    const int one = static_cast<int>(tinymind::Constants<ValueType>::one().getValue());

    srand(seed);
    static NN nn;
    nn.setLearningRate(ValueType(static_cast<Raw>(lrRaw)));
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    for (int step = 1; step <= steps; ++step)
    {
        sample<ValueType, IN, OUT>(x, t, one);
        nn.feedForward(x);
        const ValueType error = nn.calculateError(t);
        nn.trainNetwork(t);
        absorbWeights<NN, ValueType, IN, L>(nn, hash, sizes, OUT);
        if (step % 500 == 0)
        {
            std::printf("%s step=%d error=%d hash=0x%08X\n", name, step, static_cast<int>(error.getValue()), hash);
        }
    }
}

// run() with a weight-decay policy, a learning-rate schedule and an optimizer,
// an explicit learning rate in raw units, and an EarlyStopping<_, 150> fed
// every step's error (before training; the run never stops). Prints the
// learning rate at each checkpoint, and at the end the first step at which
// shouldStop() returned true, how many steps it did, and the best error.
template<typename ValueType, template<typename> class Hidden, size_t IN, typename HL, size_t L, size_t OUT,
         typename Decay, typename Schedule, typename Opt>
static void runPolicies(const char* name, const size_t (&sizes)[L], const unsigned seed, const int steps, const int lrRaw)
{
    typedef typename ValueType::FullWidthValueType Raw;
    typedef tinymind::FixedPointTransferFunctions<ValueType, RandomWeights<ValueType>, Hidden<ValueType>,
                                                  tinymind::SigmoidActivationPolicy<ValueType>, OUT,
                                                  tinymind::DefaultNetworkInitializer<ValueType>,
                                                  tinymind::MeanSquaredErrorCalculator<ValueType, OUT>,
                                                  tinymind::ZeroToleranceCalculator<ValueType>,
                                                  tinymind::NullGradientClippingPolicy<ValueType>,
                                                  Decay, Schedule, Opt> TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF> NN;
    const int one = static_cast<int>(tinymind::Constants<ValueType>::one().getValue());

    srand(seed);
    static NN nn;
    nn.setLearningRate(ValueType(static_cast<Raw>(lrRaw)));
    tinymind::EarlyStopping<ValueType, 150> stopper;
    int firstStop = 0;
    int stopCount = 0;
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    for (int step = 1; step <= steps; ++step)
    {
        sample<ValueType, IN, OUT>(x, t, one);
        nn.feedForward(x);
        const ValueType error = nn.calculateError(t);
        if (stopper.shouldStop(error))
        {
            if (firstStop == 0)
                firstStop = step;
            ++stopCount;
        }
        nn.trainNetwork(t);
        absorbWeights<NN, ValueType, IN, L>(nn, hash, sizes, OUT);
        if (step % 500 == 0)
        {
            std::printf("%s step=%d error=%d lr=%d hash=0x%08X\n", name, step, static_cast<int>(error.getValue()),
                        static_cast<int>(nn.getLearningRate().getValue()), hash);
        }
    }
    std::printf("%s stop first=%d count=%d best=%d\n", name, firstStop, stopCount,
                static_cast<int>(stopper.getBestError().getValue()));
}

int main()
{
    typedef tinymind::QValue<16, 16, true> Q16;
    typedef tinymind::QValue<8, 8, true> Q8;
    static const size_t xorSizes[2] = {4, 4};
    static const size_t deep3[3] = {8, 6, 5};
    static const size_t deep2[2] = {8, 6};
    run<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1>("q16.16 2-4-4-1", xorSizes, 1, 4000);
    run<Q8, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1>("q8.8 2-4-4-1", xorSizes, 2, 4000);
    run<Q16, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6, 5>, 3, 3>("q16.16 4-8-6-5-3", deep3, 3, 3000);
    run<Q8, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6>, 2, 3>("q8.8 4-8-6-3", deep2, 4, 3000);
    static const size_t reg2[2] = {6, 4};
    static const size_t reg1[1] = {6};
    runRegression<Q16, 3, tinymind::HiddenLayers<6, 4>, 2, 2, tinymind::GradientClipByValue<Q16> >(
        "q16.16 3-6-4-2 linear clip[-1,1]", reg2, 5, 2000);
    runRegression<Q16, 3, tinymind::HiddenLayers<6>, 1, 2, AsymmetricClip<Q16, -98304, 163840> >(
        "q16.16 3-6-2 linear clip[-1.5,2.5]", reg1, 6, 2000);

    // Adaptive optimizers. Q8.8 runs use tinymind's default hyperparameters
    // (raw 230, 255 and 1, tuned for Q8.8); Q16.16 runs use values written for
    // Q16.16: 0.9 = raw 58982, 0.999 = raw 65470, epsilon raw 16.
    typedef tinymind::AdamOptimizer<Q16, 0, 58982, 0, 65470, 0, 16> Adam16;
    typedef tinymind::RmsPropOptimizer<Q16, 0, 58982, 0, 16> RmsProp16;
    static const size_t one4[1] = {4};
    runAdaptive<Q8, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1, tinymind::AdamOptimizer<Q8> >(
        "q8.8 2-4-4-1 adam", xorSizes, 7, 2000, 16);
    runAdaptive<Q16, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6, 5>, 3, 3, Adam16>(
        "q16.16 4-8-6-5-3 adam", deep3, 8, 2000, 1 << 11);
    runAdaptive<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4>, 1, 1, Adam16>(
        "q16.16 2-4-1 adam", one4, 12, 2000, 1 << 11);
    runAdaptive<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1, RmsProp16>(
        "q16.16 2-4-4-1 rmsprop", xorSizes, 9, 2000, 1 << 10);
    runAdaptive<Q8, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6>, 2, 3, tinymind::RmsPropOptimizer<Q8> >(
        "q8.8 4-8-6-3 rmsprop", deep2, 10, 2000, 8);
    runRegression<Q16, 3, tinymind::HiddenLayers<6, 4>, 2, 2, tinymind::GradientClipByValue<Q16>, Adam16>(
        "q16.16 3-6-4-2 linear clip[-1,1] adam", reg2, 11, 2000);

    // Weight decay, step-decay schedule and early stopping. Lambda and factor
    // are raw LSBs: 655/65536 = 0.01, 52429/65536 = 0.8, 192/256 = 0.75,
    // 1311/65536 = 0.02, 62259/65536 = 0.95; the defaulted ones are
    // tinymind's (L2WeightDecay<Q16> 66, L2WeightDecay<Q8> 1, factor 0.9).
    runPolicies<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1,
                tinymind::L2WeightDecay<Q16, 0, 655>, tinymind::StepDecaySchedule<Q16, 300, 0, 52429>,
                tinymind::NullOptimizerPolicy<Q16> >("q16.16 2-4-4-1 decay schedule", xorSizes, 13, 2000, 1 << 14);
    runPolicies<Q8, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6>, 2, 3,
                tinymind::L2WeightDecay<Q8>, tinymind::StepDecaySchedule<Q8, 250, 0, 192>,
                tinymind::NullOptimizerPolicy<Q8> >("q8.8 4-8-6-3 decay schedule", deep2, 14, 2000, 64);
    runPolicies<Q16, tinymind::ReluActivationPolicy, 4, tinymind::HiddenLayers<8, 6, 5>, 3, 3,
                tinymind::L2WeightDecay<Q16>, tinymind::StepDecaySchedule<Q16, 400>, Adam16>(
        "q16.16 4-8-6-5-3 adam decay schedule", deep3, 15, 2000, 1 << 11);
    runPolicies<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4, 4>, 2, 1,
                tinymind::L2WeightDecay<Q16, 0, 655>, tinymind::StepDecaySchedule<Q16, 300, 0, 52429>,
                RmsProp16>("q16.16 2-4-4-1 rmsprop decay schedule", xorSizes, 17, 2000, 1 << 10);
    runPolicies<Q16, tinymind::TanhActivationPolicy, 2, tinymind::HiddenLayers<4>, 1, 1,
                tinymind::L2WeightDecay<Q16, 0, 1311>, tinymind::StepDecaySchedule<Q16, 200, 0, 62259>,
                tinymind::NullOptimizerPolicy<Q16> >("q16.16 2-4-1 decay schedule", one4, 16, 2000, 1 << 14);
    return 0;
}
