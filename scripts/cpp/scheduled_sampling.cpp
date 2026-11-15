// tinymind's ScheduledSampling (cpp/teacherForcing.hpp), on its own and while
// training an LSTM on its own predictions. tests/golden/sampling/stdout.txt is
// this program's output; tests/scheduled_sampling.rs does the same with
// tinymind::training::ScheduledSampling and tinymind::recurrent::LstmNetwork.
//
// Part 1 steps samplers with several decay lengths past the end of their
// window, choosing between 1 and 0 at every step: it prints how many true
// values each quarter of the run picked, the teacher-forcing ratio at those
// points, and the next rand() afterwards (which shows how many draws were
// made: none after the window). Part 2 trains a Q16.16 LSTM to predict the
// next value of a periodic sequence where each input is the true value or
// the network's previous prediction, as ScheduledSampling chooses; an FNV-1a
// hash absorbs every output and weight.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"
#include "teacherForcing.hpp"

static void sweep(const unsigned seed, const size_t total)
{
    srand(seed);
    tinymind::ScheduledSampling<int> sampler(total);
    const size_t run = total + 4;
    const size_t quarter = run / 4 + 1;
    size_t truths = 0;
    std::printf("total=%zu seed=%u", total, seed);
    for (size_t s = 0; s < run; ++s)
    {
        if (s % quarter == 0)
        {
            std::printf(" | step=%zu ratio=%.6f", sampler.getCurrentStep(), sampler.getTeacherForcingRatio());
        }
        truths += static_cast<size_t>(sampler.selectInput(1, 0));
        if (s % quarter == quarter - 1)
        {
            std::printf(" truths=%zu", truths);
            truths = 0;
        }
        sampler.step();
    }
    std::printf(" truths=%zu end=%zu next_rand=%d\n", truths, sampler.getCurrentStep(), rand());
    sampler.reset();
    std::printf("total=%zu reset step=%zu ratio=%.6f\n", total, sampler.getCurrentStep(),
                sampler.getTeacherForcingRatio());
}

typedef tinymind::QValue<16, 16, true> Q16;

struct QuarterWeights
{
    static Q16 generateRandomWeight()
    {
        const int quarter = static_cast<int>(tinymind::Constants<Q16>::one().getValue()) / 4;
        return Q16(static_cast<Q16::FullWidthValueType>((rand() % (2 * quarter + 1)) - quarter));
    }
};

typedef tinymind::FixedPointTransferFunctions<Q16, QuarterWeights, tinymind::TanhActivationPolicy<Q16>,
                                              tinymind::SigmoidActivationPolicy<Q16>, 1>
    TF;
static const size_t HID = 8;
typedef tinymind::LstmNeuralNetwork<Q16, 1, tinymind::HiddenLayers<HID>, 1, TF> NN;

static void absorb(uint32_t& hash, const Q16& v)
{
    const uint32_t raw = static_cast<uint32_t>(static_cast<int32_t>(v.getValue()));
    for (int b = 0; b < 4; ++b)
    {
        hash ^= static_cast<uint8_t>(raw >> (8 * b));
        hash *= 0x01000193u;
    }
}

static void absorbWeights(NN& nn, uint32_t& hash)
{
    for (size_t c = 0; c < 4 * HID; ++c)
        absorb(hash, nn.getInputLayerWeightForNeuronAndConnection(0, c));
    for (size_t c = 0; c < 4 * HID; ++c)
        absorb(hash, nn.getInputLayerBiasNeuronWeightForConnection(c));
    for (size_t j = 0; j < HID; ++j)
        for (size_t c = 0; c < 4 * HID; ++c)
            absorb(hash, nn.getRecurrentLayer().getWeightForNeuronAndConnection(j, c));
    for (size_t j = 0; j < HID; ++j)
        absorb(hash, nn.getHiddenLayerWeightForNeuronAndConnection(0, j, 0));
    absorb(hash, nn.getHiddenLayerBiasNeuronWeightForConnection(0, 0));
}

// One period of the sequence, raw Q16.16 values in [0.1, 0.9].
static const int SEQUENCE[12] = {6554, 15073, 26214, 39322, 50463, 57016, 58982, 52429, 41943, 30147, 19661, 11141};

static void train(const unsigned rollSeed, const size_t decaySteps, const int epochs)
{
    srand(1);
    static NN nn;
    srand(rollSeed);
    tinymind::ScheduledSampling<Q16> sampler(decaySteps);
    uint32_t hash = 0x811C9DC5u;
    int step = 0;
    int truths = 0;
    int trained = 0;
    Q16 y[1];
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        nn.resetState();
        Q16 prediction(static_cast<Q16::FullWidthValueType>(0));
        for (int t = 0; t < 24; ++t)
        {
            const Q16 truth(static_cast<Q16::FullWidthValueType>(SEQUENCE[t % 12]));
            const Q16 target(static_cast<Q16::FullWidthValueType>(SEQUENCE[(t + 1) % 12]));
            Q16 x[1];
            if (t == 0)
            {
                x[0] = truth;
            }
            else
            {
                x[0] = sampler.selectInput(truth, prediction);
                truths += (x[0] == truth) ? 1 : 0;
            }
            nn.feedForward(x);
            const Q16 error = nn.calculateError(&target);
            if (!TF::isWithinZeroTolerance(error))
            {
                nn.trainNetwork(&target);
                ++trained;
            }
            nn.getLearnedValues(y);
            prediction = y[0];
            sampler.step();
            ++step;
            absorb(hash, y[0]);
            absorbWeights(nn, hash);
            if (step % 600 == 0)
            {
                std::printf("lstm decay=%zu step=%d truths=%d trained=%d error=%d hash=0x%08X\n", decaySteps,
                            step, truths, trained, static_cast<int>(error.getValue()), hash);
            }
        }
    }
}

int main()
{
    sweep(7, 0);
    sweep(7, 1);
    sweep(3, 7);
    sweep(11, 100);
    sweep(5, 999);
    sweep(5, 1000);
    sweep(5, 1001);
    sweep(13, 4321);
    train(7, 3000, 250);
    return 0;
}
