// tinymind's QFFT1D and buildQFFTTwiddles over lengths 2 to 1024. Per length:
// the twiddle tables, then for each test signal FNV-1a hashes of forward,
// magnitudeSquared (of the input and of the spectrum), inverse and inverse
// after forward; then the same on random twiddle tables, which reach the
// products and sums that buildQFFTTwiddles' tables never produce.
// tests/golden/fft/stdout.txt is this program's output, and tests/fft_sweep.rs
// computes the same in Rust.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qfft1d.hpp"
#include "qcalibration.hpp"

static uint32_t gState;

static int16_t nextSample()
{
    gState = gState * 1664525u + 1013904223u;
    return static_cast<int16_t>(static_cast<uint16_t>(gState >> 16));
}

static uint32_t hash16(const int16_t* x, const std::size_t n, uint32_t hash)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint16_t v = static_cast<uint16_t>(x[i]);
        hash ^= v & 0xFFu;
        hash *= 0x01000193u;
        hash ^= v >> 8;
        hash *= 0x01000193u;
    }
    return hash;
}

static uint32_t hash32(const int32_t* x, const std::size_t n, uint32_t hash)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint32_t v = static_cast<uint32_t>(x[i]);
        for (int b = 0; b < 4; ++b)
        {
            hash ^= (v >> (8 * b)) & 0xFFu;
            hash *= 0x01000193u;
        }
    }
    return hash;
}

// cos(-2 pi j / N) and sin(-2 pi j / N) for j in 0..N, from the half tables:
// the second half is the negated first.
static int32_t fullCos(const int16_t* c, const std::size_t n, const std::size_t j)
{
    const std::size_t m = j % n;
    return (m < n / 2) ? c[m] : -static_cast<int32_t>(c[m - n / 2]);
}

static int32_t fullSin(const int16_t* s, const std::size_t n, const std::size_t j)
{
    const std::size_t m = j % n;
    return (m < n / 2) ? s[m] : -static_cast<int32_t>(s[m - n / 2]);
}

static const char* const kSignals[] = {"impulse", "offset-impulse", "dc", "tone", "two-tones",
                                       "random", "rails", "min", "max-min"};
static const std::size_t kNumSignals = sizeof(kSignals) / sizeof(kSignals[0]);

template<std::size_t N>
static void makeSignal(const std::size_t s, const int16_t* c, const int16_t* sn, int16_t* re, int16_t* im)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        re[i] = 0;
        im[i] = 0;
    }
    switch (s)
    {
    case 0:
        re[0] = 32767;
        break;
    case 1:
        re[3 % N] = -20000;
        im[3 % N] = 12345;
        break;
    case 2:
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = 1000;
            im[i] = -500;
        }
        break;
    case 3:
        // A cosine at bin 1.
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = static_cast<int16_t>((fullCos(c, N, i) * 20000) >> 15);
        }
        break;
    case 4:
        // A complex tone at bin 1 plus a cosine at bin 3N/8 + 1.
        for (std::size_t i = 0; i < N; ++i)
        {
            const std::size_t j = i * (3 * N / 8 + 1);
            re[i] = static_cast<int16_t>(((fullCos(c, N, i) * 12000) >> 15) + ((fullCos(c, N, j) * 9000) >> 15));
            im[i] = static_cast<int16_t>((fullSin(sn, N, i) * 12000) >> 15);
        }
        break;
    case 5:
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = nextSample();
            im[i] = nextSample();
        }
        break;
    case 6:
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = (nextSample() & 1) ? 32767 : -32768;
            im[i] = (nextSample() & 1) ? 32767 : -32768;
        }
        break;
    case 7:
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = -32768;
            im[i] = -32768;
        }
        break;
    default:
        for (std::size_t i = 0; i < N; ++i)
        {
            re[i] = 32767;
            im[i] = -32768;
        }
        break;
    }
}

template<std::size_t N>
static void runSignal(const tinymind::QFFT1D<N>& fft, const char* label, const std::size_t s,
                      const int16_t* c, const int16_t* sn)
{
    static int16_t re[N], im[N], x[N], y[N];
    static int32_t mag[N];
    makeSignal<N>(s, c, sn, x, y);

    const uint32_t hInput = hash16(y, N, hash16(x, N, 0x811C9DC5u));
    tinymind::QFFT1D<N>::magnitudeSquared(x, y, mag);
    const uint32_t hMagIn = hash32(mag, N, 0x811C9DC5u);

    std::memcpy(re, x, sizeof re);
    std::memcpy(im, y, sizeof im);
    fft.forward(re, im);
    const uint32_t hForward = hash16(im, N, hash16(re, N, 0x811C9DC5u));
    tinymind::QFFT1D<N>::magnitudeSquared(re, im, mag);
    const uint32_t hMag = hash32(mag, N, 0x811C9DC5u);
    fft.inverse(re, im);
    const uint32_t hRoundTrip = hash16(im, N, hash16(re, N, 0x811C9DC5u));

    std::memcpy(re, x, sizeof re);
    std::memcpy(im, y, sizeof im);
    fft.inverse(re, im);
    const uint32_t hInverse = hash16(im, N, hash16(re, N, 0x811C9DC5u));

    std::printf("n=%u %s signal=%s input=0x%08X mag_in=0x%08X forward=0x%08X mag=0x%08X "
                "roundtrip=0x%08X inverse=0x%08X\n",
                static_cast<unsigned>(N), label, kSignals[s], hInput, hMagIn, hForward, hMag,
                hRoundTrip, hInverse);
}

template<std::size_t N>
static void sweep()
{
    static int16_t c[N / 2], sn[N / 2];
    tinymind::buildQFFTTwiddles(N, c, sn);
    for (std::size_t k = 0; k < N / 2; k += 16)
    {
        std::printf("twiddles n=%u k=%u cos=", static_cast<unsigned>(N), static_cast<unsigned>(k));
        for (std::size_t i = k; i < k + 16 && i < N / 2; ++i)
        {
            std::printf(i == k ? "%d" : ",%d", c[i]);
        }
        std::printf(" sin=");
        for (std::size_t i = k; i < k + 16 && i < N / 2; ++i)
        {
            std::printf(i == k ? "%d" : ",%d", sn[i]);
        }
        std::printf("\n");
    }

    gState = static_cast<uint32_t>(N);
    tinymind::QFFT1D<N> fft;
    fft.twiddle_cos = c;
    fft.twiddle_sin = sn;
    for (std::size_t s = 0; s < kNumSignals; ++s)
    {
        runSignal<N>(fft, "built", s, c, sn);
    }

    // Random twiddles, -32768 included; the test signals still use the built
    // tables.
    static int16_t rc[N / 2], rs[N / 2];
    for (std::size_t i = 0; i < N / 2; ++i)
    {
        rc[i] = nextSample();
        rs[i] = nextSample();
    }
    rc[0] = -32768;
    rs[0] = -32768;
    tinymind::QFFT1D<N> randomFft;
    randomFft.twiddle_cos = rc;
    randomFft.twiddle_sin = rs;
    for (std::size_t s = 0; s < kNumSignals; ++s)
    {
        runSignal<N>(randomFft, "random-twiddles", s, c, sn);
    }
}

int main()
{
    sweep<2>();
    sweep<4>();
    sweep<8>();
    sweep<16>();
    sweep<32>();
    sweep<64>();
    sweep<128>();
    sweep<256>();
    sweep<512>();
    sweep<1024>();
    return 0;
}
