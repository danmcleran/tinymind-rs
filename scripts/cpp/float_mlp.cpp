// tinymind's NeuralNetwork in floating point (double and float), trained with
// SGD and momentum or its floating-point Adam and RMSprop, every weight hashed
// after every step.
//
// tinymind ships only FixedPointTransferFunctions; its floating-point path is
// the network templates instantiated with float or double and the activation
// specializations and transfer-function bundle that
// examples/lstm_sinusoid_float and unit_test/nn/nn_unit_test.cpp define. They
// are copied here unchanged, for double and (the same, in float) for float:
// tanh(x), 1 / (1 + exp(-x)), derivatives from the output, learning rate
// 0.15, momentum 0.5, acceleration 0.1.
//
// Five configurations with SGD: XOR 2-4-1 tanh/sigmoid in double and 2-4-4-1
// in float, 4-8-6-3 ReLU/sigmoid in double, 4-8-3 tanh/sigmoid in float, and a
// 3-6-2 ReLU/linear regression in double; then four with tinymind's
// AdamOptimizerFloat and RmsPropOptimizerFloat, attached as nn_unit_test.cpp
// attaches them (the bundle with an OptimizerPolicyType typedef); last, four
// LstmNeuralNetwork runs (see runLstm). Weights start from
// rand() / RAND_MAX - 0.5 (glibc rand, seeded per configuration). Built with
// -ffp-contract=off: a compiler that fuses multiply-adds would change the
// sums. tests/float_mlp.rs repeats every step with tinymind::NeuralNetwork and
// tinymind::recurrent::LstmNetwork over f32/f64; tests/golden/float_mlp/stdout.txt is this program's output.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "adam.hpp"
#include "rmsprop.hpp"

namespace tinymind {
    template<>
    struct TanhActivationPolicy<double>
    {
        static double activationFunction(const double& value) { return std::tanh(value); }
        static double activationFunctionDerivative(const double& value) { return (1.0 - (value * value)); }
    };

    template<>
    struct SigmoidActivationPolicy<double>
    {
        static double activationFunction(const double& value) { return (1.0 / (1.0 + std::exp(-value))); }
        static double activationFunctionDerivative(const double& value) { return (value * (1.0 - value)); }
    };

    template<>
    struct TanhActivationPolicy<float>
    {
        static float activationFunction(const float& value) { return std::tanh(value); }
        static float activationFunctionDerivative(const float& value) { return (1.0f - (value * value)); }
    };

    template<>
    struct SigmoidActivationPolicy<float>
    {
        static float activationFunction(const float& value) { return (1.0f / (1.0f + std::exp(-value))); }
        static float activationFunctionDerivative(const float& value) { return (value * (1.0f - value)); }
    };

    template<>
    struct ZeroToleranceCalculator<double>
    {
        static bool isWithinZeroTolerance(const double& value)
        {
            static const double zeroTolerance(0.004);
            static const double negativeTolerance = (-1.0 * zeroTolerance);
            return ((0 == value) || ((value < zeroTolerance) && (value > negativeTolerance)));
        }
    };

    template<>
    struct ZeroToleranceCalculator<float>
    {
        static bool isWithinZeroTolerance(const float& value)
        {
            static const float zeroTolerance(0.004f);
            static const float negativeTolerance = (-1.0f * zeroTolerance);
            return ((0 == value) || ((value < zeroTolerance) && (value > negativeTolerance)));
        }
    };

    template<>
    struct Constants<double>
    {
        static double one() { return 1.0; }
        static double negativeOne() { return -1.0; }
        static double zero() { return 0.0; }
    };

    template<>
    struct Constants<float>
    {
        static float one() { return 1.0f; }
        static float negativeOne() { return -1.0f; }
        static float zero() { return 0.0f; }
    };
}

template<
        typename ValueType,
        template<typename> class TransferFunctionRandomNumberGeneratorPolicy,
        template<typename> class TransferFunctionHiddenNeuronActivationPolicy,
        template<typename> class TransferFunctionOutputNeuronActivationPolicy,
        template<typename> class TransferFunctionGatedNeuronActivationPolicy = tinymind::NullActivationPolicy,
        template<typename> class TransferFunctionZeroTolerancePolicy = tinymind::ZeroToleranceCalculator,
        unsigned NumberOfOutputNeurons = 1>
struct FloatingPointTransferFunctions
{
    typedef ValueType TransferFunctionsValueType;
    typedef TransferFunctionRandomNumberGeneratorPolicy<ValueType> RandomNumberGeneratorPolicy;
    typedef TransferFunctionHiddenNeuronActivationPolicy<ValueType> HiddenNeuronActivationPolicy;
    typedef TransferFunctionOutputNeuronActivationPolicy<ValueType> OutputNeuronActivationPolicy;
    typedef TransferFunctionGatedNeuronActivationPolicy<ValueType> GatedNeuronActivationPolicy;
    typedef TransferFunctionZeroTolerancePolicy<ValueType> ZeroToleranceCalculatorPolicy;

    static const unsigned NumberOfTransferFunctionsOutputNeurons = NumberOfOutputNeurons;

    static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
    {
        ValueType error(0);
        for (unsigned neuron = 0; neuron < NumberOfOutputNeurons; ++neuron)
        {
            const ValueType delta = (targetValues[neuron] - outputValues[neuron]);
            error += (delta * delta);
        }
        if (NumberOfOutputNeurons > 1)
        {
            error /= NumberOfOutputNeurons;
        }
        return error;
    }

    static ValueType calculateOutputGradient(const ValueType& targetValue, const ValueType& outputValue)
    {
        const ValueType delta = targetValue - outputValue;
        return (delta * OutputNeuronActivationPolicy::activationFunctionDerivative(outputValue));
    }

    static ValueType gateActivationFunction(const ValueType& value)
    {
        return GatedNeuronActivationPolicy::activationFunction(value);
    }

    static ValueType generateRandomWeight() { return RandomNumberGeneratorPolicy::generateRandomWeight(); }

    static ValueType hiddenNeuronActivationFunction(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType hiddenNeuronActivationFunctionDerivative(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunctionDerivative(value);
    }
    static ValueType outputNeuronActivationFunction(const ValueType& value)
    {
        return OutputNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType outputNeuronActivationFunctionDerivative(const ValueType& value)
    {
        return OutputNeuronActivationPolicy::activationFunctionDerivative(value);
    }
    static ValueType neuronActivationFunction(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType neuronActivationFunctionDerivative(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunctionDerivative(value);
    }

    static ValueType initialAccelerationRate() { return ValueType(0.1); }
    static ValueType initialBiasOutputValue()  { return ValueType(1.0); }
    static ValueType initialDeltaWeight()       { return ValueType(0);   }
    static ValueType initialGradientValue()     { return ValueType(0);   }
    static ValueType initialLearningRate()      { return ValueType(0.15);}
    static ValueType initialMomentumRate()      { return ValueType(0.5); }
    static ValueType initialOutputValue()       { return ValueType(0);   }

    static bool isWithinZeroTolerance(const ValueType& value)
    {
        return ZeroToleranceCalculatorPolicy::isWithinZeroTolerance(value);
    }
    static ValueType negate(const ValueType& value) { return (-1.0 * value); }
    static ValueType noOpDeltaWeight() { return ValueType(1.0); }
    static ValueType noOpWeight()      { return ValueType(1.0); }
};

template<typename VT>
struct FloatRandomGen
{
    static VT generateRandomWeight()
    {
        return static_cast<VT>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) - 0.5);
    }
};

static uint32_t gState;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static void absorb(uint32_t& hash, const double v)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    for (int b = 0; b < 8; ++b)
    {
        hash ^= static_cast<uint8_t>(bits >> (8 * b));
        hash *= 0x01000193u;
    }
}

static void absorb(uint32_t& hash, const float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    for (int b = 0; b < 4; ++b)
    {
        hash ^= static_cast<uint8_t>(bits >> (8 * b));
        hash *= 0x01000193u;
    }
}

template<typename NN, size_t IN, size_t L>
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

// A sample: XOR of two bits for 2 inputs; for 3 inputs, a regression on
// values in [-1, 1); for 4 inputs, values in [0, 1) and a one-hot target for
// the largest of three pair sums. Every value is exact in float.
template<typename ValueType, size_t IN, size_t OUT>
static void sample(ValueType (&x)[IN], ValueType (&t)[OUT])
{
    if (IN == 2)
    {
        const int a = static_cast<int>((nextU32() >> 16) & 1u);
        const int b = static_cast<int>((nextU32() >> 16) & 1u);
        x[0] = static_cast<ValueType>(a);
        x[1] = static_cast<ValueType>(b);
        t[0] = static_cast<ValueType>(a ^ b);
        return;
    }
    int v[IN];
    for (size_t i = 0; i < IN; ++i)
    {
        v[i] = static_cast<int>((nextU32() >> 8) % 65536u);
    }
    if (IN == 3)
    {
        for (size_t i = 0; i < IN; ++i)
            x[i] = static_cast<ValueType>(v[i] - 32768) / static_cast<ValueType>(32768);
        t[0] = static_cast<ValueType>(v[0] - v[2]) / static_cast<ValueType>(65536);
        t[1 % OUT] = static_cast<ValueType>(v[1] + v[2]) / static_cast<ValueType>(131072);
        return;
    }
    for (size_t i = 0; i < IN; ++i)
        x[i] = static_cast<ValueType>(v[i]) / static_cast<ValueType>(65536);
    const int s[3] = {v[0] + v[1], v[2] + v[3], v[0] + v[3]};
    size_t best = 0;
    for (size_t k = 1; k < 3; ++k)
        if (s[k] > s[best])
            best = k;
    for (size_t o = 0; o < OUT; ++o)
        t[o] = static_cast<ValueType>(o == best ? 1 : 0);
}

static void printBits(const double v)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    std::printf(" %016llX", static_cast<unsigned long long>(bits));
}

static void printBits(const float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    std::printf(" %08X", bits);
}

// The bundle with an optimizer attached, as nn_unit_test.cpp attaches
// tinymind's floating-point Adam and RMSprop, and a learning rate of 0.01
// (0.15 saturates these networks' outputs within a few hundred steps).
template<typename Base, typename Optimizer>
struct WithOptimizer : public Base
{
    typedef Optimizer OptimizerPolicyType;
    typedef typename Base::TransferFunctionsValueType VT;
    static VT initialLearningRate() { return VT(0.01); }
};

template<typename ValueType>
struct NoOptimizer
{
};

template<typename Base, typename ValueType>
struct WithOptimizer<Base, NoOptimizer<ValueType> > : public Base
{
};

template<typename ValueType, template<typename> class Hidden, template<typename> class Output,
         size_t IN, typename HL, size_t L, size_t OUT, template<typename> class Opt = NoOptimizer>
static void run(const char* name, const size_t (&sizes)[L], const unsigned seed, const int steps)
{
    typedef FloatingPointTransferFunctions<ValueType, FloatRandomGen, Hidden, Output,
                                           tinymind::NullActivationPolicy, tinymind::ZeroToleranceCalculator,
                                           OUT> Bundle;
    typedef WithOptimizer<Bundle, Opt<ValueType> > TF;
    typedef tinymind::NeuralNetwork<ValueType, IN, HL, OUT, TF> NN;

    srand(seed);
    static NN nn;
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    for (int step = 1; step <= steps; ++step)
    {
        sample<ValueType, IN, OUT>(x, t);
        nn.feedForward(x);
        const ValueType error = nn.calculateError(t);
        nn.trainNetwork(t);
        absorbWeights<NN, IN, L>(nn, hash, sizes, OUT);
        if (step % 500 == 0)
        {
            std::printf("%s step=%d error=", name, step);
            printBits(error);
            std::printf(" hash=0x%08X\n", hash);
        }
    }
    ValueType y[OUT];
    for (int k = 0; k < 4; ++k)
    {
        sample<ValueType, IN, OUT>(x, t);
        nn.feedForward(x);
        nn.getLearnedValues(y);
        std::printf("%s final%d", name, k);
        for (size_t o = 0; o < OUT; ++o)
            printBits(y[o]);
        std::printf("\n");
    }
}

// LstmNeuralNetwork with the same bundle, on sequences of 1 to 12 steps with
// resetState() before each: inputs in [-1, 1), output 0 is 1 when the
// previous step's first input was positive, output 1 when the sign of the
// first input changed. Every step is trained unless it is a multiple of 7 or
// the error is within zero tolerance; the outputs and every weight (input,
// input bias, recurrent, hidden -> output, hidden bias) are hashed after it.
template<typename ValueType, template<typename> class Hidden, template<typename> class Output,
         size_t IN, size_t HID, size_t OUT, template<typename> class Opt = NoOptimizer>
static void runLstm(const char* name, const unsigned seed, const int steps)
{
    typedef FloatingPointTransferFunctions<ValueType, FloatRandomGen, Hidden, Output,
                                           tinymind::NullActivationPolicy, tinymind::ZeroToleranceCalculator,
                                           OUT> Bundle;
    typedef WithOptimizer<Bundle, Opt<ValueType> > TF;
    typedef tinymind::LstmNeuralNetwork<ValueType, IN, tinymind::HiddenLayers<HID>, OUT, TF> NN;

    srand(seed);
    static NN nn;
    gState = seed;
    uint32_t hash = 0x811C9DC5u;
    ValueType x[IN];
    ValueType t[OUT];
    ValueType y[OUT];
    int step = 0;
    while (step < steps)
    {
        const int length = 1 + static_cast<int>((nextU32() >> 8) % 12u);
        int previous = 0;
        nn.resetState();
        for (int k = 0; k < length && step < steps; ++k)
        {
            ++step;
            int v[IN];
            for (size_t i = 0; i < IN; ++i)
            {
                v[i] = static_cast<int>((nextU32() >> 8) % 65536u) - 32768;
                x[i] = static_cast<ValueType>(v[i]) / static_cast<ValueType>(32768);
            }
            for (size_t o = 0; o < OUT; ++o)
            {
                const int bit = (o == 0) ? (previous > 0) : ((previous > 0) != (v[0] > 0));
                t[o] = static_cast<ValueType>(bit);
            }
            previous = v[0];
            nn.feedForward(x);
            const ValueType error = nn.calculateError(t);
            if (step % 7 != 0 && !TF::isWithinZeroTolerance(error))
            {
                nn.trainNetwork(t);
            }
            nn.getLearnedValues(y);
            for (size_t o = 0; o < OUT; ++o)
                absorb(hash, y[o]);
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
            if (step % 250 == 0)
            {
                std::printf("%s step=%d error=", name, step);
                printBits(error);
                std::printf(" hash=0x%08X\n", hash);
            }
        }
    }
}

int main()
{
    const size_t four[] = {4};
    const size_t fourFour[] = {4, 4};
    const size_t eightSix[] = {8, 6};
    const size_t eight[] = {8};
    const size_t six[] = {6};
    run<double, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4>, 1, 1>("f64 2-4-1 tanh/sigmoid", four, 1, 3000);
    run<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4, 4>, 2, 1>("f32 2-4-4-1 tanh/sigmoid", fourFour, 2, 3000);
    run<double, tinymind::ReluActivationPolicy, tinymind::SigmoidActivationPolicy, 4,
        tinymind::HiddenLayers<8, 6>, 2, 3>("f64 4-8-6-3 relu/sigmoid", eightSix, 3, 2000);
    run<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 4,
        tinymind::HiddenLayers<8>, 1, 3>("f32 4-8-3 tanh/sigmoid", eight, 4, 2000);
    run<double, tinymind::ReluActivationPolicy, tinymind::LinearActivationPolicy, 3,
        tinymind::HiddenLayers<6>, 1, 2>("f64 3-6-2 relu/linear", six, 5, 2000);
    // tinymind's floating-point Adam and RMSprop.
    run<double, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4>, 1, 1, tinymind::AdamOptimizerFloat>("f64 2-4-1 tanh/sigmoid adam", four, 6, 2000);
    run<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 4,
        tinymind::HiddenLayers<8, 6>, 2, 3, tinymind::AdamOptimizerFloat>("f32 4-8-6-3 tanh/sigmoid adam", eightSix, 7, 2000);
    run<double, tinymind::ReluActivationPolicy, tinymind::LinearActivationPolicy, 3,
        tinymind::HiddenLayers<6>, 1, 2, tinymind::RmsPropOptimizerFloat>("f64 3-6-2 relu/linear rmsprop", six, 8, 2000);
    run<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2,
        tinymind::HiddenLayers<4, 4>, 2, 1, tinymind::RmsPropOptimizerFloat>("f32 2-4-4-1 tanh/sigmoid rmsprop", fourFour, 9, 2000);
    // LSTM networks.
    runLstm<double, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 3, 6, 2>(
        "f64 lstm 3-6-2 tanh/sigmoid", 10, 1500);
    runLstm<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2, 4, 2>(
        "f32 lstm 2-4-2 tanh/sigmoid", 11, 1500);
    runLstm<double, tinymind::ReluActivationPolicy, tinymind::LinearActivationPolicy, 2, 5, 1,
            tinymind::AdamOptimizerFloat>("f64 lstm 2-5-1 relu/linear adam", 12, 1500);
    runLstm<float, tinymind::TanhActivationPolicy, tinymind::SigmoidActivationPolicy, 2, 3, 2,
            tinymind::AdamOptimizerFloat>("f32 lstm 2-3-2 tanh/sigmoid adam", 13, 1500);
    return 0;
}
