// The PyTorch importer's LSTM examples (tools/import_pytorch/example_lstm.py)
// on tinymind's C++ int8 runtime. tinymind's importer has no LSTM, so
// example_lstm.py --cpp writes what it would hand the C++: float scales, the
// float LSTM biases, int8 weights and int32 dense biases (lstm.hpp,
// lstm_i8.hpp). This program builds every integer parameter from those with
// tinymind's own calibration functions (buildQLSTMParams, quantizeQLSTMBiases,
// buildQSigmoidLUT / buildQTanhLUT, buildRequantizer,
// buildQSoftmaxFixedPointParams), runs the models on QLSTMCell and QDense, and
// prints what examples/imported_lstm.rs prints, so scripts/compare-with-cpp.sh
// can byte-compare the two. A ReLU here is qreluBuffer after the requantizer.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"
#include "qdense.hpp"
#include "qlstm.hpp"
#include "qsoftmax_fixedpoint.hpp"

#include "lstm.hpp"
#include "lstm_i8.hpp"

typedef tinymind::Requantizer<int32_t, int8_t> Rq;

static uint32_t fnvBytes(uint32_t h, const uint8_t* p, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}

// Little-endian bytes of each value, as the Rust driver hashes them.
template<typename T>
static uint32_t fnv(uint32_t h, const T* v, const std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint32_t u = static_cast<uint32_t>(static_cast<int32_t>(v[i]));
        uint8_t b[sizeof(T)];
        for (std::size_t k = 0; k < sizeof(T); ++k)
        {
            b[k] = static_cast<uint8_t>(u >> (8 * k));
        }
        h = fnvBytes(h, b, sizeof(T));
    }
    return h;
}

static void printPairs(const char* label, const int32_t* m, const int32_t* s)
{
    std::printf(" %s=", label);
    for (int g = 0; g < 4; ++g)
    {
        std::printf("%s%d,%d", (g == 0) ? "" : ";", static_cast<int>(m[g]), static_cast<int>(s[g]));
    }
}

static void printRq(const char* model, const char* name, const Rq& rq)
{
    std::printf("%s %s rq=%d,%d,%d\n", model, name, static_cast<int>(rq.multiplier), static_cast<int>(rq.shift),
                static_cast<int>(rq.zero_point));
}

// Test sequence s, [step][feature], as the Rust driver computes it.
template<std::size_t T, std::size_t F>
static void input(const int s, int8_t (&x)[T][F], const float scale, const int32_t zp)
{
    for (std::size_t t = 0; t < T; ++t)
    {
        for (std::size_t k = 0; k < F; ++k)
        {
            const int i = static_cast<int>(t);
            const int j = static_cast<int>(k);
            const float v = static_cast<float>((s * 23 + i * 29 + j * 13 + s * i * 7) % 37) / 18.0f - 1.0f;
            x[t][k] = tinymind::quantize<int8_t>(v, scale, zp, -128, 127);
        }
    }
}

// The cell from the header's scales, weights and float biases.
template<typename Cell, typename C, std::size_t I, std::size_t H>
struct LstmParts
{
    int32_t biases[4 * H];
    int8_t sig[256];
    int8_t tnh[256];
    Cell cell;

    void build(const char* model, const float inScale, const int32_t inZp, const float hScale, const float cScale,
               const float lutScale, const float* wInScale, const float* wRecScale, const int8_t* wIn,
               const int8_t* wRec, const float* biasReal, const int32_t cellMin, const int32_t cellMax)
    {
        tinymind::QLSTMScales sc;
        sc.input_scale = inScale;
        sc.hidden_scale = hScale;
        sc.cell_scale = cScale;
        sc.lut_input_scale = lutScale;
        for (int g = 0; g < 4; ++g)
        {
            sc.w_input_scale[g] = wInScale[g];
            sc.w_recurrent_scale[g] = wRecScale[g];
        }
        tinymind::QLSTMParams p;
        tinymind::buildQLSTMParams(sc, p);
        tinymind::quantizeQLSTMBiases(biasReal, H, lutScale, biases);
        tinymind::buildQSigmoidLUT(lutScale, 0, tinymind::kQLstmSigmoidOutputScale, -128, sig);
        tinymind::buildQTanhLUT(lutScale, 0, tinymind::kQLstmTanhOutputScale, 0, tnh);

        cell.w_input = wIn;
        cell.w_recurrent = wRec;
        cell.biases = biases;
        cell.input_zero_point = static_cast<int8_t>(inZp);
        cell.hidden_zero_point = 0;
        cell.cell_zero_point = 0;
        for (int g = 0; g < 4; ++g)
        {
            cell.input_to_lut_multiplier[g] = p.input_to_lut_multiplier[g];
            cell.input_to_lut_shift[g] = p.input_to_lut_shift[g];
            cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
            cell.recurrent_to_lut_shift[g] = p.recurrent_to_lut_shift[g];
        }
        cell.sigmoid_lut = sig;
        cell.tanh_lut = tnh;
        cell.tanh_cell_lut = tnh;
        cell.f_times_c_multiplier = p.f_times_c_multiplier;
        cell.f_times_c_shift = p.f_times_c_shift;
        cell.i_times_g_multiplier = p.i_times_g_multiplier;
        cell.i_times_g_shift = p.i_times_g_shift;
        cell.cell_qmin = static_cast<C>(cellMin);
        cell.cell_qmax = static_cast<C>(cellMax);
        cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
        cell.cell_to_tanh_shift = p.cell_to_tanh_shift;
        cell.output_requantizer.multiplier = p.output_multiplier;
        cell.output_requantizer.shift = p.output_shift;
        cell.output_requantizer.zero_point = 0;
        cell.output_requantizer.qmin = -128;
        cell.output_requantizer.qmax = 127;

        std::printf("%s lstm", model);
        printPairs("in", cell.input_to_lut_multiplier, cell.input_to_lut_shift);
        std::printf(" rec=");
        for (int g = 0; g < 4; ++g)
        {
            std::printf("%s%d,%d", (g == 0) ? "" : ";", static_cast<int>(cell.recurrent_to_lut_multiplier[g]),
                        static_cast<int>(cell.recurrent_to_lut_shift[g]));
        }
        std::printf(" fc=%d,%d ig=%d,%d ct=%d,%d out=%d,%d cell=%d,%d\n", static_cast<int>(p.f_times_c_multiplier),
                    static_cast<int>(p.f_times_c_shift), static_cast<int>(p.i_times_g_multiplier),
                    static_cast<int>(p.i_times_g_shift), static_cast<int>(p.cell_to_tanh_multiplier),
                    static_cast<int>(p.cell_to_tanh_shift), static_cast<int>(p.output_multiplier),
                    static_cast<int>(p.output_shift), static_cast<int>(cellMin), static_cast<int>(cellMax));
        uint32_t luts = fnv(0x811C9DC5u, sig, 256);
        luts = fnv(luts, tnh, 256);
        luts = fnv(luts, tnh, 256);
        std::printf("%s lstm biases=0x%08X luts=0x%08X\n", model, fnv(0x811C9DC5u, biases, 4 * H), luts);
    }
};

static void runLstm()
{
    namespace m = example_lstm;
    typedef tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, int16_t, 3, 8> Cell;
    static LstmParts<Cell, int16_t, 3, 8> lstm;
    lstm.build("lstm", m::kinput_Scale, m::kinput_ZeroPoint, m::klstm1_Scale, m::klstm1_CellScale,
               m::klstm1_LutInputScale, m::klstm1_InputWeightScale, m::klstm1_RecurrentWeightScale,
               m::klstm1_InputWeights, m::klstm1_RecurrentWeights, m::klstm1_Bias, -32768, 32767);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 8, 4> fc1;
    fc1.weights = m::kfc1_Weights;
    fc1.biases = m::kfc1_Biases;
    fc1.input_zero_point = static_cast<int8_t>(m::klstm1_ZeroPoint);
    fc1.requantizer = tinymind::buildRequantizer<int8_t>(m::klstm1_Scale, m::kfc1_WeightScale, m::kfc1_Scale,
                                                         m::kfc1_ZeroPoint, -128, 127);
    printRq("lstm", "fc1", fc1.requantizer);
    int32_t multiplier = 0;
    int32_t leftShift = 0;
    int32_t diffMin = 0;
    tinymind::buildQSoftmaxFixedPointParams(m::kfc1_Scale, 1.0, multiplier, leftShift, diffMin);
    const tinymind::QSoftmaxFixedPoint1D<1, 4> softmax = {multiplier, leftShift, diffMin};
    std::printf("lstm softmax=%d,%d,%d\n", static_cast<int>(multiplier), static_cast<int>(leftShift),
                static_cast<int>(diffMin));

    for (int s = 0; s < 16; ++s)
    {
        int8_t x[12][3];
        input(s, x, m::kinput_Scale, m::kinput_ZeroPoint);
        int8_t h[8] = {0};
        int16_t c[8] = {0};
        uint32_t hash = 0x811C9DC5u;
        for (int t = 0; t < 12; ++t)
        {
            lstm.cell.forward(x[t], h, c);
            hash = fnv(hash, h, 8);
            hash = fnv(hash, c, 8);
        }
        int8_t logits[4];
        int8_t probs[4];
        fc1.forward(h, logits);
        softmax.forward(logits, probs);
        std::printf("lstm s=%d states=0x%08X probs=%d,%d,%d,%d\n", s, static_cast<unsigned>(hash), probs[0], probs[1],
                    probs[2], probs[3]);
    }
}

static void runLstmI8()
{
    namespace m = example_lstm_i8;
    typedef tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, 2, 5> Cell;
    static LstmParts<Cell, int8_t, 2, 5> lstm;
    lstm.build("lstm_i8", m::kinput_Scale, m::kinput_ZeroPoint, m::klstm1_Scale, m::klstm1_CellScale,
               m::klstm1_LutInputScale, m::klstm1_InputWeightScale, m::klstm1_RecurrentWeightScale,
               m::klstm1_InputWeights, m::klstm1_RecurrentWeights, m::klstm1_Bias, -128, 127);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 5, 6> fc1;
    fc1.weights = m::kfc1_Weights;
    fc1.biases = m::kfc1_Biases;
    fc1.input_zero_point = static_cast<int8_t>(m::klstm1_ZeroPoint);
    fc1.requantizer = tinymind::buildRequantizer<int8_t>(m::klstm1_Scale, m::kfc1_WeightScale, m::krelu1_Scale,
                                                         m::krelu1_ZeroPoint, -128, 127);
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 6, 1> fc2;
    fc2.weights = m::kfc2_Weights;
    fc2.biases = m::kfc2_Biases;
    fc2.input_zero_point = static_cast<int8_t>(m::krelu1_ZeroPoint);
    fc2.requantizer = tinymind::buildRequantizer<int8_t>(m::krelu1_Scale, m::kfc2_WeightScale, m::kfc2_Scale,
                                                         m::kfc2_ZeroPoint, -128, 127);
    printRq("lstm_i8", "fc1", fc1.requantizer);
    printRq("lstm_i8", "fc2", fc2.requantizer);
    int8_t lut[tinymind::kQActivationLUTSize];
    tinymind::buildQTanhLUT(m::kfc2_Scale, m::kfc2_ZeroPoint, m::kout_Scale, m::kout_ZeroPoint, lut);
    std::printf("lstm_i8 tanh lut=0x%08X\n", fnv(0x811C9DC5u, lut, 256));

    for (int s = 0; s < 16; ++s)
    {
        int8_t x[10][2];
        input(s, x, m::kinput_Scale, m::kinput_ZeroPoint);
        int8_t h[5] = {0};
        int8_t c[5] = {0};
        uint32_t hash = 0x811C9DC5u;
        for (int t = 0; t < 10; ++t)
        {
            lstm.cell.forward(x[t], h, c);
            hash = fnv(hash, h, 5);
            hash = fnv(hash, c, 5);
        }
        int8_t h1[6];
        int8_t y[1];
        fc1.forward(h, h1);
        tinymind::qreluBuffer(h1, 6, static_cast<int8_t>(m::krelu1_ZeroPoint));
        fc2.forward(h1, y);
        std::printf("lstm_i8 s=%d states=0x%08X out=%d\n", s, static_cast<unsigned>(hash),
                    tinymind::qApplyLUT(y[0], lut));
    }
}

int main()
{
    runLstm();
    runLstmI8();
    return 0;
}
