// Every tensor of tinymind's examples/kws_cortex_m_int8, for the Rust port
// (examples/kws_cortex_m_int8.rs) to match byte for byte. The example's own
// stdout starts with host timings, so this program compiles the example source
// unchanged, runs its main() with stdout captured, and prints the quantized
// input, every weight table and requantizer, every layer's output, and then
// the example's stdout from its logits report on (int8 logits against the
// float model's, which carry no timings). tests/golden/kws_int8/tensors.txt
// is its output. compare-with-cpp.sh builds it with the example's directory
// on the include path.
#define main kws_example_main
#include "kws_cortex_m_int8.cpp"
#undef main

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace {

template<typename T, std::size_t N>
void dump(const char* name, const T (&v)[N])
{
    std::printf("%s", name);
    for (std::size_t i = 0; i < N; ++i)
    {
        std::printf(" %ld", static_cast<long>(v[i]));
    }
    std::printf("\n");
}

void dumpRq(const char* name, const tinymind::Requantizer<int32_t, int8_t>& r)
{
    std::printf("%s %ld %ld %d %d %d\n", name, static_cast<long>(r.multiplier), static_cast<long>(r.shift),
                static_cast<int>(r.zero_point), static_cast<int>(r.qmin), static_cast<int>(r.qmax));
}

} // namespace

int main()
{
    std::ostringstream captured;
    std::streambuf* const saved = std::cout.rdbuf(captured.rdbuf());
    const int status = kws_example_main();
    std::cout.rdbuf(saved);
    const std::string text = captured.str();
    const std::size_t report = text.find("\nLogits (int8");

    dump("input", gInput);
    dump("w_conv1", gW_conv1);
    dumpRq("rq_conv1", gConv1.requantizer);
    dump("w_dw", gW_dw);
    for (std::size_t c = 0; c < QDwType::Channels; ++c)
    {
        dumpRq("rq_dw", gReqDw[c]);
    }
    dump("w_pw", gW_pw);
    dumpRq("rq_pw", gPw.requantizer);
    dump("w_dense", gW_dense);
    dumpRq("rq_dense", gDense.requantizer);
    dump("conv1", gBufConv1);
    dump("pool1", gBufPool1);
    dump("dw", gBufDw);
    dump("pw", gBufPw);
    dump("gap", gBufGap);
    dump("dense", gBufDense);
    std::printf("--- stdout from the logits report on\n%s", report == std::string::npos ? "(missing)\n" : text.c_str() + report + 1);
    return status;
}
