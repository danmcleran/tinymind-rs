// Every tensor of tinymind's examples/kws_cortex_m_int8, for the Rust port
// (examples/kws_cortex_m_int8.rs) to match byte for byte. The example's own
// stdout carries host timings and its logits come out all zero (the synthetic
// weights shrink the signal at every layer), so it checks little; this
// program compiles the example source unchanged, runs its main() with stdout
// silenced, and prints the quantized input, every weight table and
// requantizer, and every layer's output. tests/golden/kws_int8/tensors.txt is
// its output. Build it from the example's directory (compare-with-cpp.sh
// does) so the example's includes resolve.
#define main kws_example_main
#include "kws_cortex_m_int8.cpp"
#undef main

#include <cstdio>
#include <iostream>

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
    std::streambuf* const saved = std::cout.rdbuf(nullptr);
    const int status = kws_example_main();
    std::cout.rdbuf(saved);
    std::cout.clear();

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
    return status;
}
