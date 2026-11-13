// tinymind's DenseHead and PrototypeHead (cpp/densehead.hpp,
// cpp/prototypehead.hpp) driven by an integer LCG. Every value is drawn as a
// whole number of 2^-16 steps (2^-8 for Q8.8), so the Q formats, float and
// double all start from exactly the same numbers. Per step the program prints
// what the head returned and FNV-1a hashes of every parameter and every output
// (raw bits for the Q formats, IEEE bits for the floats). The ranges are
// chosen to reach the softmax's zero below -6, the exp table's interpolation,
// wrapping sums in Q8.8 and wrapping squared distances. No float softmax: glibc
// changed expf between 2.39 and 2.41, and the Rust side calls the same expf.
// tests/golden/head/stdout.txt is this program's output, and
// tests/head_sweep.rs computes the same in Rust.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qformat.hpp"
#include "densehead.hpp"
#include "prototypehead.hpp"

static uint32_t gState;

static uint32_t next()
{
    gState = gState * 1664525u + 1013904223u;
    return gState >> 8;
}

// A whole number in [-range, range].
static int32_t draw(const int32_t range)
{
    return static_cast<int32_t>(next() % static_cast<uint32_t>(2 * range + 1)) - range;
}

static uint32_t fnv(uint32_t hash, const uint64_t bits)
{
    for (int b = 0; b < 8; ++b)
    {
        hash ^= static_cast<uint32_t>((bits >> (8 * b)) & 0xFFu);
        hash *= 0x01000193u;
    }
    return hash;
}

static const uint32_t FNV_OFFSET = 0x811C9DC5u;

typedef tinymind::QValue<16, 16, true> Q16;
typedef tinymind::QValue<8, 8, true> Q8;

// One value type: how to build a value from raw steps, its bits, its name.
template<typename V>
struct VT;

template<>
struct VT<Q16>
{
    static const int32_t ONE = 1 << 16;
    static Q16 make(const int32_t raw) { Q16 v; v.setValue(raw); return v; }
    static uint64_t bits(const Q16& v) { return static_cast<uint64_t>(static_cast<int64_t>(v.getValue())); }
    static const char* name() { return "q16.16"; }
};

template<>
struct VT<Q8>
{
    static const int32_t ONE = 1 << 8;
    static Q8 make(const int32_t raw) { Q8 v; v.setValue(static_cast<int16_t>(raw)); return v; }
    static uint64_t bits(const Q8& v) { return static_cast<uint64_t>(static_cast<int64_t>(v.getValue())); }
    static const char* name() { return "q8.8"; }
};

template<>
struct VT<double>
{
    static const int32_t ONE = 1 << 16;
    static double make(const int32_t raw) { return static_cast<double>(raw) / 65536.0; }
    static uint64_t bits(const double& v) { uint64_t b; std::memcpy(&b, &v, sizeof(b)); return b; }
    static const char* name() { return "f64"; }
};

template<>
struct VT<float>
{
    static const int32_t ONE = 1 << 16;
    static float make(const int32_t raw) { return static_cast<float>(raw) / 65536.0f; }
    static uint64_t bits(const float& v) { uint32_t b; std::memcpy(&b, &v, sizeof(b)); return b; }
    static const char* name() { return "f32"; }
};

// A value in [-units, units], in raw steps.
template<typename V>
static V value(const int32_t units)
{
    return VT<V>::make(draw(units * VT<V>::ONE));
}

template<typename V>
static uint32_t hashValues(V const* const x, const size_t n, uint32_t hash)
{
    for (size_t i = 0; i < n; ++i)
    {
        hash = fnv(hash, VT<V>::bits(x[i]));
    }
    return hash;
}

template<typename V, int Units>
struct WeightDraw
{
    static V generateRandomWeight()
    {
        return value<V>(Units);
    }
};

template<typename HeadType, size_t IN, size_t OUT, typename V>
static uint32_t hashHead(const HeadType& head)
{
    uint32_t hash = FNV_OFFSET;
    for (size_t o = 0; o < OUT; ++o)
    {
        for (size_t i = 0; i < IN; ++i)
        {
            hash = fnv(hash, VT<V>::bits(head.getWeight(o, i)));
        }
        hash = fnv(hash, VT<V>::bits(head.getBias(o)));
    }
    return hash;
}

// lr is in raw steps; inputUnits bounds the embedding.
template<typename V, size_t IN, size_t OUT, typename Loss>
static void denseRun(const char* const loss, const uint32_t seed, const int32_t inputUnits, const int32_t lrRaw)
{
    typedef tinymind::DenseHead<V, IN, OUT, Loss> HeadType;
    gState = seed;
    HeadType head;
    std::printf("dense %s %s %zux%zu zero %08x\n", VT<V>::name(), loss, IN, OUT, hashHead<HeadType, IN, OUT, V>(head));
    head.template initializeWeights<WeightDraw<V, 1> >();
    std::printf("dense init %08x\n", hashHead<HeadType, IN, OUT, V>(head));

    const V rate = VT<V>::make(lrRaw);
    for (int step = 0; step < 64; ++step)
    {
        V input[IN];
        for (size_t i = 0; i < IN; ++i)
        {
            input[i] = value<V>(inputUnits);
        }
        V inputDeltas[IN];
        for (size_t i = 0; i < IN; ++i)
        {
            inputDeltas[i] = VT<V>::make(0);
        }
        const uint32_t kind = next() % 4;
        size_t result = 0;
        if (kind == 3)
        {
            // The pieces on their own, with a target that is not one-hot.
            V targets[OUT];
            for (size_t o = 0; o < OUT; ++o)
            {
                targets[o] = value<V>(1);
            }
            V logits[OUT];
            V deltas[OUT];
            head.forward(input, logits);
            head.outputDeltas(logits, targets, deltas);
            head.backward(deltas, inputDeltas);
            head.update(deltas, input, rate);
            result = head.predict(input);
            std::printf("step %d parts logits %08x deltas %08x", step, hashValues(logits, OUT, FNV_OFFSET), hashValues(deltas, OUT, FNV_OFFSET));
        }
        else
        {
            // Class OUT (out of range) gives an all-zero target.
            const size_t classIndex = next() % (OUT + 1);
            result = head.trainClass(input, classIndex, rate, (kind == 0) ? nullptr : inputDeltas);
            std::printf("step %d class %zu", step, classIndex);
        }
        std::printf(" pred %zu head %08x in %08x\n", result, hashHead<HeadType, IN, OUT, V>(head), hashValues(inputDeltas, IN, FNV_OFFSET));
    }

    HeadType copy;
    V weights[IN * OUT];
    V biases[OUT];
    for (size_t o = 0; o < OUT; ++o)
    {
        for (size_t i = 0; i < IN; ++i)
        {
            weights[o * IN + i] = head.getWeight(o, i);
        }
        biases[o] = head.getBias(o);
    }
    copy.loadWeights(weights, biases);
    head.clear();
    V probe[IN];
    for (size_t i = 0; i < IN; ++i)
    {
        probe[i] = value<V>(inputUnits);
    }
    // A cleared head ties every logit: the first class wins.
    std::printf("dense loaded %08x cleared %08x pred %zu\n", hashHead<HeadType, IN, OUT, V>(copy), hashHead<HeadType, IN, OUT, V>(head), head.predict(probe));
}

template<typename V, size_t N>
static void softmaxRun(const uint32_t seed, const int32_t units)
{
    gState = seed;
    for (int k = 0; k < 32; ++k)
    {
        V logits[N];
        V p[N];
        for (size_t o = 0; o < N; ++o)
        {
            logits[o] = value<V>(units);
        }
        tinymind::SoftmaxCrossEntropyHeadLoss<V>::probabilities(logits, p, N);
        std::printf("softmax %s %zu #%d %08x\n", VT<V>::name(), N, k, hashValues(p, N, FNV_OFFSET));
    }
}

template<typename V, size_t IN, size_t C, typename Distance>
static void prototypeRun(const char* const distance, const uint32_t seed, const int32_t inputUnits)
{
    typedef tinymind::PrototypeHead<V, IN, C, Distance> HeadType;
    gState = seed;
    HeadType head;
    std::printf("prototype %s %s %zux%zu\n", VT<V>::name(), distance, IN, C);

    for (int step = 0; step < 64; ++step)
    {
        V input[IN];
        for (size_t i = 0; i < IN; ++i)
        {
            input[i] = value<V>(inputUnits);
        }
        const uint32_t op = next() % 16;
        const size_t classIndex = next() % C;
        if (op < 9)
        {
            head.learn(input, classIndex);
            std::printf("step %d learn %zu", step, classIndex);
        }
        else if (op == 9)
        {
            static const uint16_t caps[4] = {0, 1, 3, 65535};
            const uint16_t cap = caps[next() % 4];
            head.setMaximumCount(cap);
            std::printf("step %d cap %u", step, static_cast<unsigned>(cap));
        }
        else if (op == 10)
        {
            const uint16_t count = static_cast<uint16_t>(next() % 6);
            head.setPrototype(classIndex, input, count);
            std::printf("step %d set %zu %u", step, classIndex, static_cast<unsigned>(count));
        }
        else if (op == 11)
        {
            head.forget(classIndex);
            std::printf("step %d forget %zu", step, classIndex);
        }
        else if ((op == 12) && ((next() % 4) == 0))
        {
            head.clear();
            std::printf("step %d clear", step);
        }
        else
        {
            std::printf("step %d query", step);
        }

        V query[IN];
        for (size_t i = 0; i < IN; ++i)
        {
            query[i] = value<V>(inputUnits);
        }
        uint32_t distances = FNV_OFFSET;
        uint32_t prototypes = FNV_OFFSET;
        for (size_t c = 0; c < C; ++c)
        {
            distances = fnv(distances, VT<V>::bits(head.distance(query, c)));
            for (size_t i = 0; i < IN; ++i)
            {
                prototypes = fnv(prototypes, VT<V>::bits(head.getPrototype(c, i)));
            }
        }
        std::printf(" pred %zu max %u counts", head.predict(query), static_cast<unsigned>(head.getMaximumCount()));
        for (size_t c = 0; c < C; ++c)
        {
            std::printf(" %u", static_cast<unsigned>(head.getCount(c)));
        }
        std::printf(" d %08x p %08x\n", distances, prototypes);
    }

    // Two classes with the same prototype tie: the first one wins.
    V shared[IN];
    for (size_t i = 0; i < IN; ++i)
    {
        shared[i] = value<V>(inputUnits);
    }
    head.clear();
    head.setPrototype(C - 1, shared, 1);
    head.setPrototype(1, shared, 2);
    std::printf("tie pred %zu\n", head.predict(shared));
}

// The on-board workload (bench/src/head.rs and the C++ board images): Q16.16
// embeddings of 32 values for 6 classes, 5 samples each, a shipped dense head
// and the prototypes of all 30 samples. Each line is the FNV-1a hash, over the
// 30 samples in turn, of the state after one operation on a fresh copy, so
// the boards' hashes can be compared with these.
static const size_t B_IN = 32;
static const size_t B_OUT = 6;
static const size_t B_N = 30;
typedef tinymind::DenseHead<Q16, B_IN, B_OUT> BenchHead;
typedef tinymind::PrototypeHead<Q16, B_IN, B_OUT> BenchPrototypes;

static uint32_t hashDense(const BenchHead& head, uint32_t hash)
{
    for (size_t o = 0; o < B_OUT; ++o)
    {
        for (size_t i = 0; i < B_IN; ++i)
        {
            hash = fnv(hash, VT<Q16>::bits(head.getWeight(o, i)));
        }
        hash = fnv(hash, VT<Q16>::bits(head.getBias(o)));
    }
    return hash;
}

static uint32_t hashPrototypes(const BenchPrototypes& p, uint32_t hash)
{
    for (size_t c = 0; c < B_OUT; ++c)
    {
        for (size_t i = 0; i < B_IN; ++i)
        {
            hash = fnv(hash, VT<Q16>::bits(p.getPrototype(c, i)));
        }
        hash = fnv(hash, p.getCount(c));
    }
    return hash;
}

static void benchRun()
{
    gState = 2026u;
    Q16 x[B_N][B_IN];
    size_t label[B_N];
    for (size_t s = 0; s < B_N; ++s)
    {
        label[s] = s % B_OUT;
        for (size_t i = 0; i < B_IN; ++i)
        {
            const int32_t base = static_cast<int32_t>((label[s] * 5 + i * 3) % 8) * (VT<Q16>::ONE / 4);
            x[s][i] = VT<Q16>::make(base + draw(VT<Q16>::ONE / 4));
        }
    }
    BenchHead shipped;
    shipped.initializeWeights<WeightDraw<Q16, 1> >();
    const Q16 rate = VT<Q16>::make(3277); // 0.05
    BenchPrototypes learned;
    for (size_t s = 0; s < B_N; ++s)
    {
        learned.learn(x[s], label[s]);
    }

    uint32_t step = FNV_OFFSET;
    uint32_t predict = FNV_OFFSET;
    uint32_t learn = FNV_OFFSET;
    uint32_t nearest = FNV_OFFSET;
    for (size_t s = 0; s < B_N; ++s)
    {
        BenchHead head = shipped;
        head.trainClass(x[s], label[s], rate);
        step = hashDense(head, step);
        predict = fnv(predict, shipped.predict(x[s]));
        BenchPrototypes p = learned;
        p.learn(x[s], label[s]);
        learn = hashPrototypes(p, learn);
        nearest = fnv(nearest, learned.predict(x[s]));
    }
    BenchHead adapted = shipped;
    for (int epoch = 0; epoch < 30; ++epoch)
    {
        for (size_t s = 0; s < B_N; ++s)
        {
            adapted.trainClass(x[s], label[s], rate);
        }
    }
    size_t correct = 0;
    for (size_t s = 0; s < B_N; ++s)
    {
        correct += (adapted.predict(x[s]) == label[s]) ? 1u : 0u;
    }
    std::printf("bench head_step %08x\n", step);
    std::printf("bench head_predict %08x\n", predict);
    std::printf("bench proto_learn %08x\n", learn);
    std::printf("bench proto_predict %08x\n", nearest);
    std::printf("bench head_adapt %08x correct %zu\n", hashDense(adapted, FNV_OFFSET), correct);
}

int main()
{
    using tinymind::SoftmaxCrossEntropyHeadLoss;
    using tinymind::SquaredErrorHeadLoss;
    using tinymind::SquaredEuclideanDistance;
    using tinymind::ManhattanDistance;

    denseRun<Q16, 8, 4, SoftmaxCrossEntropyHeadLoss<Q16> >("sce", 1u, 3, 1 << 13);
    denseRun<Q16, 8, 4, SquaredErrorHeadLoss<Q16> >("se", 2u, 3, 1 << 11);
    denseRun<Q16, 3, 5, SoftmaxCrossEntropyHeadLoss<Q16> >("sce", 3u, 40, 1 << 14);
    denseRun<Q8, 4, 3, SoftmaxCrossEntropyHeadLoss<Q8> >("sce", 4u, 2, 1 << 5);
    denseRun<Q8, 4, 3, SquaredErrorHeadLoss<Q8> >("se", 5u, 8, 1 << 6);
    denseRun<Q8, 1, 1, SoftmaxCrossEntropyHeadLoss<Q8> >("sce", 6u, 2, 1 << 5);
    denseRun<double, 8, 4, SoftmaxCrossEntropyHeadLoss<double> >("sce", 7u, 3, 1 << 13);
    denseRun<double, 8, 4, SquaredErrorHeadLoss<double> >("se", 8u, 3, 1 << 11);
    denseRun<float, 8, 4, SquaredErrorHeadLoss<float> >("se", 9u, 3, 1 << 11);

    softmaxRun<Q16, 5>(10u, 10);
    softmaxRun<Q8, 5>(11u, 10);
    softmaxRun<double, 5>(12u, 10);

    prototypeRun<Q16, 8, 4, SquaredEuclideanDistance<Q16> >("euclid", 20u, 3);
    prototypeRun<Q16, 8, 4, ManhattanDistance<Q16> >("manhattan", 21u, 50);
    prototypeRun<Q16, 4, 3, SquaredEuclideanDistance<Q16> >("euclid", 22u, 100);
    prototypeRun<Q8, 4, 3, SquaredEuclideanDistance<Q8> >("euclid", 23u, 2);
    prototypeRun<Q8, 4, 3, ManhattanDistance<Q8> >("manhattan", 24u, 20);
    prototypeRun<double, 8, 4, SquaredEuclideanDistance<double> >("euclid", 25u, 3);
    prototypeRun<float, 8, 4, ManhattanDistance<float> >("manhattan", 26u, 3);

    benchRun();

    return 0;
}
