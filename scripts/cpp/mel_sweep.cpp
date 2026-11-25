// tinymind's QLogMelSpectrogram, QMfcc and the host builders
// (buildQHannWindow, buildQMelFilterbank, buildQDctMatrix; cpp/qmel.hpp) over
// frame lengths 4 to 1024, band counts that leave some bands empty, output Q
// formats 0, 8 and 14, and log floors from -32768 to 32767. Per configuration:
// hashes of the built tables, then for each test signal (silence, a one-LSB
// impulse, noise of a few LSB, full-scale noise and rails, a sawtooth, a quiet
// square wave, DC, all -32768, a pre-emphasis step) with pre-emphasis off and
// on (and, for some configurations, at -32768 and at 1/32768, which leaves a
// silent frame with one tiny sample), the log-mel bands and MFCCs. Then the same signals through random
// tables (window, band starts and lengths, weights), which reach sums the
// built tables never produce. Integer test signals only, so no libm.
// tests/golden/mel/stdout.txt is this program's output, and
// tests/mel_sweep.rs computes the same in Rust.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define TINYMIND_ENABLE_FLOAT 1
#define TINYMIND_ENABLE_STD 1
#include "qcalibration.hpp"
#include "qmel.hpp"

static uint32_t gState = 7;

static uint32_t next()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static int16_t nextI16()
{
    return static_cast<int16_t>(static_cast<uint16_t>(next() >> 16));
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

static const int kSignals = 10;

template<std::size_t N>
static int16_t makeSignal(int s, int16_t* x)
{
    int16_t previous = 0;
    for (std::size_t n = 0; n < N; ++n)
    {
        int32_t v = 0;
        switch (s)
        {
        case 0: v = 0; break;
        case 1: v = (n == N / 2) ? 1 : 0; break;
        case 2: v = static_cast<int32_t>(next() >> 30) - 2; break;
        case 3: v = nextI16(); break;
        case 4: v = ((next() >> 31) != 0u) ? 32767 : -32768; break;
        case 5:
        {
            const std::size_t p = (N >= 8) ? N / 8 : 2;
            v = static_cast<int32_t>((n % p) * 6000 / p) - 3000;
            break;
        }
        case 6: v = ((n / 4) % 2 == 0) ? 3 : -3; break;
        case 7: v = 1000; break;
        case 8: v = -32768; break;
        default: v = 20000; previous = -32768; break;
        }
        x[n] = static_cast<int16_t>(v);
    }
    return previous;
}

template<std::size_t N, std::size_t Bands, int Frac, std::size_t Coeffs>
static void runFrames(const char* name, const tinymind::QLogMelSpectrogram<N, Bands, Frac>& mel,
                      const tinymind::QMfcc<Bands, Coeffs>& mfcc, bool extremePreemphasis)
{
    static tinymind::QMelWorkspace<N> workspace;
    tinymind::QLogMelSpectrogram<N, Bands, Frac> layer = mel;
    const int variants = extremePreemphasis ? 4 : 2;
    for (int s = 0; s < kSignals; ++s)
    {
        int16_t frame[N];
        const int16_t signalPrevious = makeSignal<N>(s, frame);
        for (int v = 0; v < variants; ++v)
        {
            int16_t previous = signalPrevious;
            if (v == 0)
            {
                layer.preemphasis = 0;
            }
            else if (v == 1)
            {
                layer.preemphasis = 31785;
                previous = (s == 9) ? signalPrevious : nextI16();
            }
            else if (v == 2)
            {
                layer.preemphasis = -32768;
                previous = 32767;
            }
            else
            {
                // On silence only sample 0 is nonzero, -w[0]: below 2^14 the
                // block scale shifts left.
                layer.preemphasis = 1;
                previous = 1;
            }
            int16_t out[Bands];
            int32_t coeffs[Coeffs];
            layer.forward(frame, previous, out, workspace);
            mfcc.forward(out, coeffs);
            std::printf("mel %s signal=%d pre=%d prev=%d out=", name, s, layer.preemphasis, previous);
            for (std::size_t m = 0; m < ((Bands < 4) ? Bands : 4); ++m)
            {
                std::printf("%s%d", (m == 0) ? "" : ",", out[m]);
            }
            std::printf(" hash=%08x mfcc=%d,%d hash=%08x\n", hashBytes(out, sizeof(out)), coeffs[0],
                        coeffs[(Coeffs > 1) ? 1 : 0], hashBytes(coeffs, sizeof(coeffs)));
        }
    }
}

template<std::size_t N, std::size_t Bands, int Frac, std::size_t Coeffs>
static void built(const char* name, double sampleRate, double fMin, double fMax, int16_t logFloor, bool periodic,
                  bool extremePreemphasis)
{
    static int16_t window[N];
    static int16_t twiddleCos[N / 2];
    static int16_t twiddleSin[N / 2];
    static uint16_t bandStart[Bands];
    static uint16_t bandLength[Bands];
    static uint16_t weights[2 * (N / 2 + 1)];
    static int16_t dct[Coeffs * Bands];
    tinymind::buildQHannWindow(N, window, periodic);
    tinymind::buildQFFTTwiddles(N, twiddleCos, twiddleSin);
    const std::size_t used = tinymind::buildQMelFilterbank(N, sampleRate, Bands, fMin, fMax, bandStart, bandLength,
                                                           weights, 2 * (N / 2 + 1));
    const std::size_t tooSmall = (used == 0) ? 0 : tinymind::buildQMelFilterbank(
        N, sampleRate, Bands, fMin, fMax, bandStart, bandLength, weights, used - 1);
    // The short call may have overwritten the tables; rebuild them.
    tinymind::buildQMelFilterbank(N, sampleRate, Bands, fMin, fMax, bandStart, bandLength, weights,
                                  2 * (N / 2 + 1));
    tinymind::buildQDctMatrix(Bands, Coeffs, dct);
    std::size_t empty = 0;
    for (std::size_t m = 0; m < Bands; ++m)
    {
        empty += (bandLength[m] == 0) ? 1 : 0;
    }
    std::printf("tables %s window=%08x first=%d,%d,%d used=%zu short=%zu empty=%zu start=%08x length=%08x "
                "weights=%08x dct=%08x dct0=%d,%d\n",
                name, hashBytes(window, sizeof(window)), window[0], window[1], window[N / 2], used, tooSmall, empty,
                hashBytes(bandStart, sizeof(bandStart)), hashBytes(bandLength, sizeof(bandLength)),
                hashBytes(weights, used * sizeof(uint16_t)), hashBytes(dct, sizeof(dct)), dct[0],
                dct[(Coeffs > 1) ? Bands : 0]);
    const tinymind::QLogMelSpectrogram<N, Bands, Frac> mel = {window, twiddleCos, twiddleSin, bandStart, bandLength,
                                                              weights, 0, logFloor};
    const tinymind::QMfcc<Bands, Coeffs> mfcc = {dct};
    runFrames<N, Bands, Frac, Coeffs>(name, mel, mfcc, extremePreemphasis);
}

template<std::size_t N, std::size_t Bands, int Frac>
static void randomTables(const char* name)
{
    static int16_t window[N];
    static int16_t twiddleCos[N / 2];
    static int16_t twiddleSin[N / 2];
    static uint16_t bandStart[Bands];
    static uint16_t bandLength[Bands];
    static uint16_t weights[Bands * (N / 2 + 1)];
    static int16_t dct[Bands * Bands];
    tinymind::buildQFFTTwiddles(N, twiddleCos, twiddleSin);
    for (std::size_t n = 0; n < N; ++n)
    {
        window[n] = nextI16();
    }
    window[0] = (N == 64) ? static_cast<int16_t>(-32768) : static_cast<int16_t>(100);
    std::size_t used = 0;
    const std::size_t bins = N / 2 + 1;
    for (std::size_t m = 0; m < Bands; ++m)
    {
        bandStart[m] = static_cast<uint16_t>((next() >> 16) % bins);
        bandLength[m] = static_cast<uint16_t>((next() >> 16) % (bins - bandStart[m] + 1));
        for (std::size_t j = 0; j < bandLength[m]; ++j)
        {
            weights[used++] = (m == 0) ? static_cast<uint16_t>(65535) : static_cast<uint16_t>(next() >> 16);
        }
    }
    for (std::size_t i = 0; i < Bands * Bands; ++i)
    {
        dct[i] = nextI16();
    }
    std::printf("tables %s random used=%zu start=%08x length=%08x\n", name, used,
                hashBytes(bandStart, sizeof(bandStart)), hashBytes(bandLength, sizeof(bandLength)));
    const tinymind::QLogMelSpectrogram<N, Bands, Frac> mel = {window, twiddleCos, twiddleSin, bandStart, bandLength,
                                                              weights, 0, static_cast<int16_t>(-32768)};
    const tinymind::QMfcc<Bands, Bands> mfcc = {dct};
    runFrames<N, Bands, Frac, Bands>(name, mel, mfcc, true);
}

int main()
{
    built<4, 2, 8, 1>("n4", 8.0, 0.0, 4.0, -32768, true, true);
    built<64, 10, 8, 5>("n64", 8000.0, 20.0, 4000.0, -1500, true, false);
    built<128, 64, 8, 13>("n128-narrow", 16000.0, 0.0, 8000.0, -32768, true, false);
    built<256, 32, 8, 13>("n256", 16000.0, 20.0, 8000.0, -2048, true, true);
    built<256, 32, 0, 32>("n256-q0", 16000.0, 20.0, 8000.0, -128, false, false);
    built<512, 26, 14, 13>("n512-q14", 44100.0, 30.0, 11000.0, -32768, true, false);
    built<1024, 40, 8, 20>("n1024", 48000.0, 0.0, 24000.0, 32767, true, false);
    randomTables<16, 5, 8>("rand16");
    randomTables<64, 7, 14>("rand64");
    return 0;
}
