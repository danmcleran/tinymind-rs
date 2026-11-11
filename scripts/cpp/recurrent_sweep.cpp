// tinymind's int8 recurrent cells, QLSTMCell (int8 and int16 cell state) and
// QGRUCell, with their calibration: buildQLSTMParams, buildQGRUParams,
// quantizeQLSTMBiases, quantizeQGRUBiases and the sigmoid/tanh LUT builders.
//
// For each shape, two calibrated trials (random float weights and biases in
// [-0.5, 0.5), random exact scales and zero points, with the hidden, cell and
// LUT input scales covering about +-1 to 2, +-4 to 8 and +-8 to 16; weights
// quantized symmetrically per gate, LUTs built in the cells' fixed output
// grids) and two direct trials (random int8 weights, int32 biases, multipliers, shifts,
// LUTs, zero points and output ranges, qmin > qmax included). Shifts stay at
// -3 or above so no i32 sum overflows: everything stays inside the C++
// contract (the program is UBSan clean). Trial 0 starts from the zero-point
// state, trial 1 from a random one. Each trial runs 32 steps on random inputs.
//
// Each line prints the parameters and hashes (FNV-1a) the weights, biases,
// LUTs and the hidden and cell state after every step. The draws come in a
// fixed order, which tests/recurrent_int8_sweep.rs repeats with the Rust cells.
// tests/golden/recurrent_int8/stdout.txt is this program's output.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qlstm.hpp"
#include "qgru.hpp"
#include "include/qcalibration.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

static const int kSteps = 32;

static uint32_t gState = 0x5EED0005u;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static uint32_t draw(const uint32_t n)
{
    return (nextU32() >> 8) % n;
}

static int8_t randI8()
{
    return static_cast<int8_t>(static_cast<int32_t>(draw(256)) - 128);
}

static int16_t randI16()
{
    return static_cast<int16_t>(static_cast<int32_t>(draw(65536)) - 32768);
}

// (1 + d) * 2^-(16 + base + e), e in 0..3: exact in float.
static float randScale(const int base)
{
    const float m = static_cast<float>(1 + draw(1u << 16));
    return std::ldexp(m, -16 - base - static_cast<int>(draw(4)));
}

// (2^16 + d) * 2^-(17 + base): exact in float, in [2^-(base + 1), 2^-base).
static float narrowScale(const int base)
{
    const float m = static_cast<float>((1u << 16) + draw(1u << 16));
    return std::ldexp(m, -17 - base);
}

// (d - 32768) * 2^-16: exact in float, in [-0.5, 0.5).
static float randReal()
{
    return std::ldexp(static_cast<float>(static_cast<int32_t>(draw(65536)) - 32768), -16);
}

static int32_t randMultiplier()
{
    return static_cast<int32_t>((1u << 30) + (draw(1u << 24) << 6));
}

// A quarter of the shifts in -3..1 (left shifts and the requantizer's
// two-step path), the rest in lo..lo + n - 1.
static int32_t randShift(const int32_t lo, const uint32_t n)
{
    if (draw(4) == 0)
    {
        return static_cast<int32_t>(draw(5)) - 3;
    }
    return lo + static_cast<int32_t>(draw(n));
}

static uint32_t bits(const float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

static uint32_t fnvByte(uint32_t h, const uint8_t b)
{
    h ^= b;
    return h * 0x01000193u;
}

static uint32_t fnv(uint32_t h, const int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        h = fnvByte(h, static_cast<uint8_t>(p[i]));
    }
    return h;
}

// Little-endian bytes.
static uint32_t fnv(uint32_t h, const int16_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint16_t u = static_cast<uint16_t>(p[i]);
        h = fnvByte(h, static_cast<uint8_t>(u & 0xFF));
        h = fnvByte(h, static_cast<uint8_t>(u >> 8));
    }
    return h;
}

static uint32_t fnv(uint32_t h, const int32_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint32_t u = static_cast<uint32_t>(p[i]);
        for (int k = 0; k < 4; ++k)
        {
            h = fnvByte(h, static_cast<uint8_t>((u >> (8 * k)) & 0xFF));
        }
    }
    return h;
}

static void fill(int8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        p[i] = randI8();
    }
}

static void fillReal(float* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        p[i] = randReal();
    }
}

// absmax / 127, as tinymind's unit tests scale a gate's weights; 1/127 for an
// all-zero gate.
static float absmaxScale(const float* p, const std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float a = std::fabs(p[i]);
        if (a > m)
        {
            m = a;
        }
    }
    return (m > 0.0f) ? m / 127.0f : 1.0f / 127.0f;
}

// tinymind's quantizeSymmetricWeights (unit_test/quantization): round half
// away from zero, clamp to [-127, 127].
static void quantizeWeights(const float* src, int8_t* dst, const std::size_t n, const float scale)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        long c = std::lround(static_cast<double>(src[i]) / static_cast<double>(scale));
        if (c < -127) c = -127;
        if (c > 127) c = 127;
        dst[i] = static_cast<int8_t>(c);
    }
}

static void randLut(int8_t* lut)
{
    fill(lut, 256);
}

template<typename C> struct CellLimits;
template<> struct CellLimits<int8_t>
{
    static const int32_t min = -128;
    static const int32_t max = 127;
    static int8_t any() { return randI8(); }
    // Calibrated cell zero point: small.
    static int8_t calibratedZeroPoint() { return static_cast<int8_t>(randI8() / 8); }
    static int8_t directZeroPoint() { return randI8(); }
};
template<> struct CellLimits<int16_t>
{
    static const int32_t min = -32768;
    static const int32_t max = 32767;
    static int16_t any() { return randI16(); }
    static int16_t calibratedZeroPoint()
    {
        return static_cast<int16_t>(static_cast<int32_t>(draw(2001)) - 1000);
    }
    // Within 1000 of 0, so c - zp stays below 2^16 and no product overflows.
    static int16_t directZeroPoint() { return calibratedZeroPoint(); }
};

static void printRescales(const char* label, const int32_t* m, const int32_t* s, const std::size_t n)
{
    std::printf(" %s=", label);
    for (std::size_t g = 0; g < n; ++g)
    {
        std::printf("%s%d,%d", (g == 0) ? "" : ";", m[g], s[g]);
    }
}

template<typename C, std::size_t I, std::size_t H>
static void lstm(const char* name, const bool calibrated, const int trial)
{
    typedef tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, C, I, H> Cell;
    typedef CellLimits<C> L;
    const bool wide = sizeof(C) == 2;

    int8_t w_in[4 * H * I], w_rec[4 * H * H];
    int32_t bias[4 * H];
    int8_t sig[256], tnh[256], tnh_c[256];
    Cell cell;
    cell.w_input = w_in;
    cell.w_recurrent = w_rec;
    cell.biases = bias;
    cell.sigmoid_lut = sig;
    cell.tanh_lut = tnh;
    cell.tanh_cell_lut = tnh_c;

    std::printf("lstm %s %s %d", name, calibrated ? "calibrated" : "direct", trial);
    if (calibrated)
    {
        float wf_in[4 * H * I], wf_rec[4 * H * H], bf[4 * H];
        fillReal(wf_in, 4 * H * I);
        fillReal(wf_rec, 4 * H * H);
        fillReal(bf, 4 * H);
        tinymind::QLSTMScales sc;
        sc.input_scale = randScale(6);
        sc.hidden_scale = narrowScale(6);
        sc.cell_scale = narrowScale(wide ? 12 : 4);
        sc.lut_input_scale = narrowScale(3);
        for (std::size_t g = 0; g < 4; ++g)
        {
            sc.w_input_scale[g] = absmaxScale(wf_in + g * H * I, H * I);
            sc.w_recurrent_scale[g] = absmaxScale(wf_rec + g * H * H, H * H);
        }
        tinymind::QLSTMParams p;
        tinymind::buildQLSTMParams(sc, p);
        tinymind::quantizeQLSTMBiases(bf, H, sc.lut_input_scale, bias);
        for (std::size_t g = 0; g < 4; ++g)
        {
            quantizeWeights(wf_in + g * H * I, w_in + g * H * I, H * I, sc.w_input_scale[g]);
            quantizeWeights(wf_rec + g * H * H, w_rec + g * H * H, H * H, sc.w_recurrent_scale[g]);
        }
        tinymind::buildQSigmoidLUT(sc.lut_input_scale, 0, 1.0f / 256.0f, -128, sig);
        tinymind::buildQTanhLUT(sc.lut_input_scale, 0, 1.0f / 128.0f, 0, tnh);
        tinymind::buildQTanhLUT(sc.lut_input_scale, 0, 1.0f / 128.0f, 0, tnh_c);

        cell.input_zero_point = randI8();
        cell.hidden_zero_point = static_cast<int8_t>(randI8() / 8);
        cell.cell_zero_point = L::calibratedZeroPoint();
        for (std::size_t g = 0; g < 4; ++g)
        {
            cell.input_to_lut_multiplier[g] = p.input_to_lut_multiplier[g];
            cell.input_to_lut_shift[g] = p.input_to_lut_shift[g];
            cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
            cell.recurrent_to_lut_shift[g] = p.recurrent_to_lut_shift[g];
        }
        cell.f_times_c_multiplier = p.f_times_c_multiplier;
        cell.f_times_c_shift = p.f_times_c_shift;
        cell.i_times_g_multiplier = p.i_times_g_multiplier;
        cell.i_times_g_shift = p.i_times_g_shift;
        cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
        cell.cell_to_tanh_shift = p.cell_to_tanh_shift;
        cell.cell_qmin = static_cast<C>((draw(2) == 0) ? L::min + 1 : L::min);
        cell.cell_qmax = static_cast<C>(L::max);
        cell.output_requantizer.multiplier = p.output_multiplier;
        cell.output_requantizer.shift = p.output_shift;
        cell.output_requantizer.zero_point = cell.hidden_zero_point;
        cell.output_requantizer.qmin = -128;
        cell.output_requantizer.qmax = 127;
        std::printf(" scales=0x%08X,0x%08X,0x%08X,0x%08X", bits(sc.input_scale), bits(sc.hidden_scale),
                    bits(sc.cell_scale), bits(sc.lut_input_scale));
    }
    else
    {
        fill(w_in, 4 * H * I);
        fill(w_rec, 4 * H * H);
        for (std::size_t k = 0; k < 4 * H; ++k)
        {
            bias[k] = static_cast<int32_t>(draw(8193)) - 4096;
        }
        randLut(sig);
        randLut(tnh);
        randLut(tnh_c);
        cell.input_zero_point = randI8();
        cell.hidden_zero_point = randI8();
        cell.cell_zero_point = L::directZeroPoint();
        for (std::size_t g = 0; g < 4; ++g)
        {
            cell.input_to_lut_multiplier[g] = randMultiplier();
            cell.input_to_lut_shift[g] = randShift(6, 8);
            cell.recurrent_to_lut_multiplier[g] = randMultiplier();
            cell.recurrent_to_lut_shift[g] = randShift(6, 8);
        }
        cell.f_times_c_multiplier = randMultiplier();
        cell.f_times_c_shift = randShift(6, 8);
        cell.i_times_g_multiplier = randMultiplier();
        cell.i_times_g_shift = randShift(6, 8);
        cell.cell_to_tanh_multiplier = randMultiplier();
        cell.cell_to_tanh_shift = randShift(wide ? 6 : 0, 10);
        const uint32_t kind = draw(3);
        if (kind == 0)
        {
            cell.cell_qmin = static_cast<C>(L::min);
            cell.cell_qmax = static_cast<C>(L::max);
        }
        else if (kind == 1)
        {
            cell.cell_qmin = static_cast<C>(L::min + 1);
            cell.cell_qmax = static_cast<C>(L::max);
        }
        else
        {
            cell.cell_qmin = L::any(); // may exceed qmax: the result is then qmax
            cell.cell_qmax = L::any();
        }
        cell.output_requantizer.multiplier = randMultiplier();
        cell.output_requantizer.shift = randShift(6, 8);
        cell.output_requantizer.zero_point = randI8();
        cell.output_requantizer.qmin = randI8();
        cell.output_requantizer.qmax = randI8();
        if (draw(2) == 0)
        {
            cell.output_requantizer.qmin = -128;
            cell.output_requantizer.qmax = 127;
        }
    }

    printRescales("in", cell.input_to_lut_multiplier, cell.input_to_lut_shift, 4);
    printRescales("rec", cell.recurrent_to_lut_multiplier, cell.recurrent_to_lut_shift, 4);
    std::printf(" fc=%d,%d ig=%d,%d ct=%d,%d rq=%d,%d,%d,%d,%d", cell.f_times_c_multiplier, cell.f_times_c_shift,
                cell.i_times_g_multiplier, cell.i_times_g_shift, cell.cell_to_tanh_multiplier,
                cell.cell_to_tanh_shift, cell.output_requantizer.multiplier, cell.output_requantizer.shift,
                cell.output_requantizer.zero_point, cell.output_requantizer.qmin, cell.output_requantizer.qmax);
    std::printf(" zp=%d,%d,%d cell=%d,%d", cell.input_zero_point, cell.hidden_zero_point,
                static_cast<int32_t>(cell.cell_zero_point), static_cast<int32_t>(cell.cell_qmin),
                static_cast<int32_t>(cell.cell_qmax));
    std::printf(" w=0x%08X b=0x%08X lut=0x%08X", fnv(fnv(0x811C9DC5u, w_in, 4 * H * I), w_rec, 4 * H * H),
                fnv(0x811C9DC5u, bias, 4 * H),
                fnv(fnv(fnv(0x811C9DC5u, sig, 256), tnh, 256), tnh_c, 256));

    int8_t h[H];
    C c[H];
    for (std::size_t k = 0; k < H; ++k)
    {
        if (trial == 0)
        {
            h[k] = cell.hidden_zero_point;
            c[k] = cell.cell_zero_point;
        }
        else
        {
            h[k] = randI8();
            c[k] = L::any();
        }
    }
    uint32_t hash = 0x811C9DC5u;
    for (int t = 0; t < kSteps; ++t)
    {
        int8_t x[I];
        fill(x, I);
        cell.forward(x, h, c);
        hash = fnv(hash, h, H);
        hash = fnv(hash, c, H);
    }
    std::printf(" h=");
    for (std::size_t k = 0; k < H; ++k)
    {
        std::printf("%s%d", (k == 0) ? "" : ",", h[k]);
    }
    std::printf(" hash=0x%08X\n", hash);
}

template<std::size_t I, std::size_t H>
static void gru(const char* name, const bool calibrated, const int trial)
{
    typedef tinymind::QGRUCell<int8_t, int8_t, int32_t, int8_t, int8_t, I, H> Cell;

    int8_t w_in[3 * H * I], w_rec[3 * H * H];
    int32_t bias[3 * H];
    int8_t sig[256], tnh[256];
    Cell cell;
    cell.w_input = w_in;
    cell.w_recurrent = w_rec;
    cell.biases = bias;
    cell.sigmoid_lut = sig;
    cell.tanh_lut = tnh;

    std::printf("gru %s %s %d", name, calibrated ? "calibrated" : "direct", trial);
    if (calibrated)
    {
        float wf_in[3 * H * I], wf_rec[3 * H * H], bf[3 * H];
        fillReal(wf_in, 3 * H * I);
        fillReal(wf_rec, 3 * H * H);
        fillReal(bf, 3 * H);
        tinymind::QGRUScales sc;
        sc.input_scale = randScale(6);
        sc.hidden_scale = narrowScale(6);
        sc.lut_input_scale = narrowScale(3);
        for (std::size_t g = 0; g < 3; ++g)
        {
            sc.w_input_scale[g] = absmaxScale(wf_in + g * H * I, H * I);
            sc.w_recurrent_scale[g] = absmaxScale(wf_rec + g * H * H, H * H);
        }
        tinymind::QGRUParams p;
        tinymind::buildQGRUParams(sc, p);
        tinymind::quantizeQGRUBiases(bf, H, sc.lut_input_scale, bias);
        for (std::size_t g = 0; g < 3; ++g)
        {
            quantizeWeights(wf_in + g * H * I, w_in + g * H * I, H * I, sc.w_input_scale[g]);
            quantizeWeights(wf_rec + g * H * H, w_rec + g * H * H, H * H, sc.w_recurrent_scale[g]);
        }
        tinymind::buildQSigmoidLUT(sc.lut_input_scale, 0, 1.0f / 256.0f, -128, sig);
        tinymind::buildQTanhLUT(sc.lut_input_scale, 0, 1.0f / 128.0f, 0, tnh);

        cell.input_zero_point = randI8();
        cell.hidden_zero_point = static_cast<int8_t>(randI8() / 8);
        for (std::size_t g = 0; g < 3; ++g)
        {
            cell.input_to_lut_multiplier[g] = p.input_to_lut_multiplier[g];
            cell.input_to_lut_shift[g] = p.input_to_lut_shift[g];
            cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
            cell.recurrent_to_lut_shift[g] = p.recurrent_to_lut_shift[g];
        }
        cell.r_times_h_multiplier = p.r_times_h_multiplier;
        cell.r_times_h_shift = p.r_times_h_shift;
        cell.one_minus_z_times_n_multiplier = p.one_minus_z_times_n_multiplier;
        cell.one_minus_z_times_n_shift = p.one_minus_z_times_n_shift;
        cell.z_times_h_multiplier = p.z_times_h_multiplier;
        cell.z_times_h_shift = p.z_times_h_shift;
        cell.output_qmin = static_cast<int8_t>((draw(2) == 0) ? -127 : -128);
        cell.output_qmax = 127;
        std::printf(" scales=0x%08X,0x%08X,0x%08X", bits(sc.input_scale), bits(sc.hidden_scale),
                    bits(sc.lut_input_scale));
    }
    else
    {
        fill(w_in, 3 * H * I);
        fill(w_rec, 3 * H * H);
        for (std::size_t k = 0; k < 3 * H; ++k)
        {
            bias[k] = static_cast<int32_t>(draw(8193)) - 4096;
        }
        randLut(sig);
        randLut(tnh);
        cell.input_zero_point = randI8();
        cell.hidden_zero_point = randI8();
        for (std::size_t g = 0; g < 3; ++g)
        {
            cell.input_to_lut_multiplier[g] = randMultiplier();
            cell.input_to_lut_shift[g] = randShift(6, 8);
            cell.recurrent_to_lut_multiplier[g] = randMultiplier();
            cell.recurrent_to_lut_shift[g] = randShift(6, 8);
        }
        cell.r_times_h_multiplier = randMultiplier();
        cell.r_times_h_shift = randShift(6, 8);
        cell.one_minus_z_times_n_multiplier = randMultiplier();
        cell.one_minus_z_times_n_shift = randShift(6, 8);
        cell.z_times_h_multiplier = randMultiplier();
        cell.z_times_h_shift = randShift(6, 8);
        cell.output_qmin = randI8(); // may exceed qmax: the result is then qmax
        cell.output_qmax = randI8();
        if (draw(2) == 0)
        {
            cell.output_qmin = -128;
            cell.output_qmax = 127;
        }
    }

    printRescales("in", cell.input_to_lut_multiplier, cell.input_to_lut_shift, 3);
    printRescales("rec", cell.recurrent_to_lut_multiplier, cell.recurrent_to_lut_shift, 3);
    std::printf(" rh=%d,%d zn=%d,%d zh=%d,%d", cell.r_times_h_multiplier, cell.r_times_h_shift,
                cell.one_minus_z_times_n_multiplier, cell.one_minus_z_times_n_shift, cell.z_times_h_multiplier,
                cell.z_times_h_shift);
    std::printf(" zp=%d,%d out=%d,%d", cell.input_zero_point, cell.hidden_zero_point, cell.output_qmin,
                cell.output_qmax);
    std::printf(" w=0x%08X b=0x%08X lut=0x%08X", fnv(fnv(0x811C9DC5u, w_in, 3 * H * I), w_rec, 3 * H * H),
                fnv(0x811C9DC5u, bias, 3 * H), fnv(fnv(0x811C9DC5u, sig, 256), tnh, 256));

    int8_t h[H];
    for (std::size_t k = 0; k < H; ++k)
    {
        h[k] = (trial == 0) ? cell.hidden_zero_point : randI8();
    }
    uint32_t hash = 0x811C9DC5u;
    for (int t = 0; t < kSteps; ++t)
    {
        int8_t x[I];
        fill(x, I);
        cell.forward(x, h);
        hash = fnv(hash, h, H);
    }
    std::printf(" h=");
    for (std::size_t k = 0; k < H; ++k)
    {
        std::printf("%s%d", (k == 0) ? "" : ",", h[k]);
    }
    std::printf(" hash=0x%08X\n", hash);
}

template<typename C, std::size_t I, std::size_t H>
static void lstmShape(const char* name)
{
    lstm<C, I, H>(name, true, 0);
    lstm<C, I, H>(name, true, 1);
    lstm<C, I, H>(name, false, 0);
    lstm<C, I, H>(name, false, 1);
}

template<std::size_t I, std::size_t H>
static void gruShape(const char* name)
{
    gru<I, H>(name, true, 0);
    gru<I, H>(name, true, 1);
    gru<I, H>(name, false, 0);
    gru<I, H>(name, false, 1);
}

int main()
{
    lstmShape<int8_t, 3, 4>("3x4_i8");
    lstmShape<int16_t, 3, 4>("3x4_i16");
    lstmShape<int8_t, 1, 1>("1x1_i8");
    lstmShape<int16_t, 2, 3>("2x3_i16");
    lstmShape<int8_t, 8, 16>("8x16_i8");
    lstmShape<int16_t, 16, 8>("16x8_i16");
    lstmShape<int8_t, 5, 7>("5x7_i8");

    gruShape<3, 4>("3x4");
    gruShape<1, 1>("1x1");
    gruShape<2, 3>("2x3");
    gruShape<8, 16>("8x16");
    gruShape<16, 8>("16x8");
    gruShape<5, 7>("5x7");
    return 0;
}
