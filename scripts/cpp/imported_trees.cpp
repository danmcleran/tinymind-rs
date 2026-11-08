// The imported-trees example on tinymind's C++ runtime: the same two
// scikit-learn models for the UCI Wine data, imported by tinymind's own
// importer (apps/import_trees) into gbdt.hpp and forest.hpp, run on the
// held-out samples (trees_data.hpp) and on random inputs. Every evaluation
// policy must agree; the output must be byte-identical to
// `cargo run --example imported_trees`. scripts/compare-with-cpp.sh writes the
// headers and builds this; tests/golden/imported_trees/stdout.txt is its output.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "gbdt.hpp"
#include "forest.hpp"
#include "trees_data.hpp"

static uint32_t gState;

static uint32_t draw(const uint32_t n)
{
    gState = gState * 1664525u + 1013904223u;
    return (gState >> 8) % n;
}

static void fnv(uint32_t& hash, const int32_t v)
{
    const uint32_t u = static_cast<uint32_t>(v);
    for (int b = 0; b < 4; ++b)
    {
        hash ^= (u >> (8 * b)) & 0xFFu;
        hash *= 0x01000193u;
    }
}

#define RUN_MODEL(NS, TYPE_NAME)                                                                   \
    {                                                                                              \
        typedef NS::InputType In;                                                                  \
        std::printf("model %s: %zu trees, %zu %s inputs, %zu outputs\n", #NS,                    \
                    static_cast<std::size_t>(NS::NumTrees), static_cast<std::size_t>(NS::NumFeatures), \
                    TYPE_NAME, static_cast<std::size_t>(NS::NumOutputs));                          \
        std::size_t correct = 0;                                                                   \
        for (std::size_t s = 0; s < kNumSamples; ++s)                                              \
        {                                                                                          \
            In q[NS::NumFeatures];                                                                 \
            NS::quantizeFeatures(kTestX[s], q);                                                    \
            int32_t raw[NS::NumOutputs], c1[NS::NumOutputs], c2[NS::NumOutputs];                   \
            NS::forward<tinymind::QTreeInlineEval>(q, raw);                                        \
            NS::forward<tinymind::QTreeCompactEval>(q, c1);                                        \
            NS::forward<tinymind::QTreeWalkEval>(q, c2);                                           \
            const std::size_t cls = NS::predict<tinymind::QTreeInlineEval>(q);                     \
            for (std::size_t k = 0; k < NS::NumOutputs; ++k)                                       \
                if (raw[k] != c1[k] || raw[k] != c2[k]) std::abort();                              \
            if (cls != NS::predict<tinymind::QTreeCompactEval>(q)                                  \
                || cls != NS::predict<tinymind::QTreeWalkEval>(q)) std::abort();                   \
            correct += (cls == kTestY[s]) ? 1u : 0u;                                               \
            std::printf("sample %2zu: q=[", s);                                                    \
            for (std::size_t f = 0; f < NS::NumFeatures; ++f)                                      \
                std::printf(f ? " %d" : "%d", static_cast<int>(q[f]));                             \
            std::printf("] raw=[");                                                                \
            for (std::size_t k = 0; k < NS::NumOutputs; ++k)                                       \
                std::printf(k ? " %ld" : "%ld", static_cast<long>(raw[k]));                        \
            std::printf("] class=%zu label=%zu\n", cls, kTestY[s]);                                \
        }                                                                                          \
        std::printf("held-out accuracy: %zu/%zu\n", correct, kNumSamples);                         \
        gState = 1;                                                                                \
        const int32_t lo = std::numeric_limits<In>::min();                                         \
        const uint32_t span = 1u << (8 * sizeof(In));                                              \
        uint32_t hash = 0x811C9DC5u;                                                               \
        for (int i = 0; i < 256; ++i)                                                              \
        {                                                                                          \
            In q[NS::NumFeatures];                                                                 \
            for (std::size_t f = 0; f < NS::NumFeatures; ++f)                                      \
                q[f] = static_cast<In>(lo + static_cast<int32_t>(draw(span)));                     \
            int32_t raw[NS::NumOutputs], c2[NS::NumOutputs];                                       \
            NS::forward<tinymind::QTreeInlineEval>(q, raw);                                        \
            NS::forward<tinymind::QTreeWalkEval>(q, c2);                                           \
            for (std::size_t k = 0; k < NS::NumOutputs; ++k)                                       \
            {                                                                                      \
                if (raw[k] != c2[k]) std::abort();                                                 \
                fnv(hash, raw[k]);                                                                 \
            }                                                                                      \
            fnv(hash, static_cast<int32_t>(NS::predict<tinymind::QTreeWalkEval>(q)));              \
        }                                                                                          \
        std::printf("random inputs: 256, hash=0x%08X\n", static_cast<unsigned>(hash));             \
    }

int main()
{
    RUN_MODEL(gbdt, "int8")
    RUN_MODEL(forest, "int16")
    return 0;
}
