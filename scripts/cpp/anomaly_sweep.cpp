// tinymind's QKMeansAnomaly, QKMeansAnomalyTrainer, QDiagGaussianAnomaly and
// buildQDiagGaussianAnomaly (cpp/qanomaly.hpp) on random tables and sample
// streams: scores on direct random tables (weighted, unweighted, unused
// clusters, distances past 2^32); the trainer's state and scores after
// learning clustered streams with different step caps and with feature
// weighting on and off, including the edge cases (calibrating before any
// sample, fewer samples than clusters, learning after calibration); the
// Gaussian score on random tables and on tables built from random float fits.
// tests/golden/qanomaly/stdout.txt is this program's output, and
// tests/qanomaly_sweep.rs computes the same in Rust.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define TINYMIND_ENABLE_FLOAT 1
#define TINYMIND_ENABLE_STD 1
#include "qanomaly.hpp"

static uint32_t gState = 1;

static uint32_t next()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static int8_t nextI8()
{
    return static_cast<int8_t>(static_cast<uint8_t>(next() >> 24));
}

static uint16_t nextU16()
{
    return static_cast<uint16_t>(next() >> 16);
}

static uint32_t nextU32()
{
    const uint32_t high = next() >> 16;
    return (high << 16) | (next() >> 16);
}

// Uniform in [0, 1).
static double nextUnit()
{
    return static_cast<double>(next() >> 8) / 16777216.0;
}

static int8_t clampI8(int32_t v)
{
    return static_cast<int8_t>((v < -128) ? -128 : ((v > 127) ? 127 : v));
}

static uint32_t hashBytes(const void* data, std::size_t n, uint32_t hash = 0x811C9DC5u)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i)
    {
        hash ^= p[i];
        hash *= 0x01000193u;
    }
    return hash;
}

// A sample: extreme, or random.
template<std::size_t F>
static void drawSample(int8_t* x, int kind)
{
    for (std::size_t f = 0; f < F; ++f)
    {
        x[f] = (kind == 0) ? static_cast<int8_t>(-128) : ((kind == 1) ? static_cast<int8_t>(127) : nextI8());
    }
}

template<std::size_t F, std::size_t K>
static void directKMeans(const char* name)
{
    int8_t centroids[K][F];
    uint32_t inv[K];
    uint16_t weights[F];
    for (int trial = 0; trial < 4; ++trial)
    {
        for (std::size_t k = 0; k < K; ++k)
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                centroids[k][f] = nextI8();
            }
            const uint32_t r = next() >> 29;
            // Unused, the largest, small (so a distance past 2^32 can still
            // score below INT32_MAX), or any.
            inv[k] = (r == 0) ? 0u : ((r == 1) ? UINT32_MAX : ((r == 2) ? (next() >> 22) : nextU32()));
        }
        for (std::size_t f = 0; f < F; ++f)
        {
            weights[f] = (trial == 3) ? static_cast<uint16_t>(65535) : nextU16();
        }
        const bool weighted = (trial % 2) == 1;
        const tinymind::QKMeansAnomaly<F, K> model = {&centroids[0][0], inv, weighted ? weights : nullptr};
        std::printf("kmeans %s trial=%d weighted=%d scores=", name, trial, weighted ? 1 : 0);
        for (int s = 0; s < 12; ++s)
        {
            int8_t x[F];
            drawSample<F>(x, s);
            std::printf("%s%d", (s == 0) ? "" : ",", model.score(x));
        }
        std::printf("\n");
    }
}

// Clustered stream: one of K centers (drawn once per run) plus noise whose
// spread differs per feature, so the feature weights differ. Feature 1 never
// moves (variance 0: the largest weight); with 512 or more features feature 0
// swings across the whole range and the rest barely move (a weight below 0.5
// for feature 0, clamped to 1). A
// constant stream (every center equal, no noise) makes every distance a tie.
template<std::size_t F, std::size_t K>
struct Stream
{
    int8_t centers[K][F];
    int spread[F];

    void init(bool constant)
    {
        const int8_t common = static_cast<int8_t>(static_cast<int32_t>(nextI8()) / 2);
        for (std::size_t k = 0; k < K; ++k)
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                centers[k][f] = constant ? common : static_cast<int8_t>(static_cast<int32_t>(nextI8()) / 2);
            }
        }
        for (std::size_t f = 0; f < F; ++f)
        {
            if (constant || f == 1)
            {
                spread[f] = 0;
            }
            else if (F >= 512)
            {
                spread[f] = (f == 0) ? 127 : 1;
            }
            else
            {
                spread[f] = 1 + static_cast<int>(next() >> 28);
            }
        }
    }

    void draw(int8_t* x)
    {
        const std::size_t k = (next() >> 16) % K;
        for (std::size_t f = 0; f < F; ++f)
        {
            const int32_t noise = static_cast<int32_t>((next() >> 16) % static_cast<uint32_t>(2 * spread[f] + 1)) - spread[f];
            x[f] = clampI8(static_cast<int32_t>(centers[k][f]) + noise);
        }
    }
};

template<std::size_t F, std::size_t K>
static void printTrainer(const char* name, int run, const tinymind::QKMeansAnomalyTrainer<F, K>& t)
{
    std::printf("trainer %s run=%d seeded=%zu counts=", name, run, t.seededClusters());
    for (std::size_t k = 0; k < K; ++k)
    {
        std::printf("%s%u", (k == 0) ? "" : ",", static_cast<unsigned>(t.learnCount(k)));
    }
    std::printf(" calibration=");
    for (std::size_t k = 0; k < K; ++k)
    {
        std::printf("%s%u", (k == 0) ? "" : ",", static_cast<unsigned>(t.calibrationCount(k)));
    }
    uint32_t hash = 0x811C9DC5u;
    for (std::size_t k = 0; k < K; ++k)
    {
        hash = hashBytes(t.centroid(k), F, hash);
    }
    std::printf(" centroids=%08x weights=", hash);
    for (std::size_t f = 0; f < ((F > 8) ? 8 : F); ++f)
    {
        std::printf("%s%u", (f == 0) ? "" : ",", static_cast<unsigned>(t.featureWeight(f)));
    }
    uint32_t weightHash = 0x811C9DC5u;
    for (std::size_t f = 0; f < F; ++f)
    {
        const uint16_t w = t.featureWeight(f);
        weightHash = hashBytes(&w, sizeof(w), weightHash);
    }
    std::printf(" weight_hash=%08x", weightHash);
    const tinymind::QKMeansAnomaly<F, K> model = t.model();
    std::printf(" inv=");
    for (std::size_t k = 0; k < K; ++k)
    {
        std::printf("%s%u", (k == 0) ? "" : ",", model.inv_radius_sq[k]);
    }
    std::printf("\n");
}

template<std::size_t F, std::size_t K>
static void trainers(const char* name)
{
    static const uint16_t caps[4] = {65535, 0, 5, 40};
    for (int run = 0; run < 9; ++run)
    {
        Stream<F, K> stream;
        stream.init(run == 8);
        static tinymind::QKMeansAnomalyTrainer<F, K> t;
        t.reset();
        if (run != 0)
        {
            t.setMaximumCount(caps[run % 4]);
        }
        t.setFeatureWeighting(run < 4 || run >= 6);
        const int learnCount = (run == 5) ? static_cast<int>(K / 2) : 50 + 37 * run;
        int8_t x[F];
        if (run == 7)
        {
            // Calibrating before any sample is learned changes nothing.
            stream.draw(x);
            t.calibrate(x);
            printTrainer<F, K>(name, run, t);
            t.reset();
        }
        for (int i = 0; i < learnCount; ++i)
        {
            stream.draw(x);
            t.learn(x);
        }
        for (int i = 0; i < 40 + 3 * run; ++i)
        {
            stream.draw(x);
            t.calibrate(x);
        }
        // Ignored once frozen.
        stream.draw(x);
        t.learn(x);
        printTrainer<F, K>(name, run, t);
        const tinymind::QKMeansAnomaly<F, K> model = t.model();
        std::printf("trainer %s run=%d scores=", name, run);
        for (int s = 0; s < 16; ++s)
        {
            if (s < 8)
            {
                stream.draw(x);
            }
            else
            {
                drawSample<F>(x, s - 8);
            }
            std::printf("%s%d", (s == 0) ? "" : ",", model.score(x));
        }
        std::printf("\n");
    }
}

template<std::size_t F, std::size_t K>
static void gaussians(const char* name)
{
    int8_t means[K][F];
    uint32_t precision[K][F];
    int32_t bias[K];
    for (int trial = 0; trial < 6; ++trial)
    {
        if (trial < 3)
        {
            for (std::size_t k = 0; k < K; ++k)
            {
                for (std::size_t f = 0; f < F; ++f)
                {
                    means[k][f] = nextI8();
                    precision[k][f] = (trial == 2) ? UINT32_MAX : nextU32();
                }
                bias[k] = (trial == 1) ? static_cast<int32_t>(nextU32()) : static_cast<int32_t>(next() >> 20) - 2048;
            }
            if (trial == 2)
            {
                bias[0] = INT32_MIN;
            }
        }
        else
        {
            double fm[K][F];
            double fv[K][F];
            double fw[K];
            const float scale = static_cast<float>(0.01 + nextUnit());
            const int32_t zp = static_cast<int32_t>(next() >> 26) - 32;
            double total = 0.0;
            for (std::size_t k = 0; k < K; ++k)
            {
                for (std::size_t f = 0; f < F; ++f)
                {
                    fm[k][f] = (nextUnit() - 0.5) * 300.0 * scale;
                    // Some variances far below the floor, some large.
                    const double u = nextUnit();
                    fv[k][f] = (u < 0.2) ? 1e-9 : u * u * 400.0 * static_cast<double>(scale) * scale;
                }
                fw[k] = 0.05 + nextUnit();
                total += fw[k];
            }
            for (std::size_t k = 0; k < K; ++k)
            {
                fw[k] /= total;
            }
            const double floorLsb2 = (trial == 5) ? 0.0001 : 0.25;
            tinymind::buildQDiagGaussianAnomaly(K, F, &fm[0][0], &fv[0][0], fw, scale, zp, floorLsb2,
                                                &means[0][0], &precision[0][0], bias);
            std::printf("gaussian %s trial=%d built means=%08x precision=%08x bias=", name, trial,
                        hashBytes(means, sizeof(means)), hashBytes(precision, sizeof(precision)));
            for (std::size_t k = 0; k < K; ++k)
            {
                std::printf("%s%d", (k == 0) ? "" : ",", bias[k]);
            }
            std::printf("\n");
        }
        const tinymind::QDiagGaussianAnomaly<F, K> model = {&means[0][0], &precision[0][0], bias};
        std::printf("gaussian %s trial=%d scores=", name, trial);
        for (int s = 0; s < 12; ++s)
        {
            int8_t x[F];
            drawSample<F>(x, s);
            std::printf("%s%d", (s == 0) ? "" : ",", model.score(x));
        }
        std::printf("\n");
    }
}

int main()
{
    directKMeans<1, 1>("1x1");
    directKMeans<4, 3>("4x3");
    directKMeans<7, 2>("7x2");
    directKMeans<32, 8>("32x8");
    trainers<1, 1>("1x1");
    trainers<4, 3>("4x3");
    trainers<7, 2>("7x2");
    trainers<32, 8>("32x8");
    trainers<600, 2>("600x2");
    gaussians<1, 1>("1x1");
    gaussians<4, 3>("4x3");
    gaussians<32, 2>("32x2");
    return 0;
}
