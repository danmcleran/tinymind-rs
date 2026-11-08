// tinymind's QDecisionTree and QGBDT over random ensembles: int8, uint8 and
// int16 thresholds and inputs, eight trees of depth up to 5 sharing a 512-node
// pool, three classes, with and without base scores. Leaf values are small, so
// classes often tie, and half of the inputs sit on or next to a split
// threshold. Each model's line hashes every score, prediction and single-tree
// value. tests/golden/tree/stdout.txt is this program's output, and
// tests/tree_sweep.rs computes the same with tinymind::tree.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "qtree.hpp"

static uint32_t gState;

static uint32_t nextU32()
{
    gState = gState * 1664525u + 1013904223u;
    return gState;
}

static uint32_t draw(const uint32_t n)
{
    return (nextU32() >> 8) % n;
}

template<typename T> struct Range;
template<> struct Range<int8_t>   { static constexpr int32_t lo = -128;   static constexpr uint32_t span = 256; };
template<> struct Range<uint8_t>  { static constexpr int32_t lo = 0;      static constexpr uint32_t span = 256; };
template<> struct Range<int16_t>  { static constexpr int32_t lo = -32768; static constexpr uint32_t span = 65536; };

template<typename T>
static T randT()
{
    return static_cast<T>(static_cast<int32_t>(draw(Range<T>::span)) + Range<T>::lo);
}

constexpr std::size_t NODES = 512;
constexpr std::size_t TREES = 8;
constexpr std::size_t CLASSES = 3;
constexpr std::size_t FEATURES = 6;
constexpr std::size_t INPUTS = 64;
constexpr std::size_t MODELS = 24;
constexpr int MAX_DEPTH = 5;

template<typename T>
static int16_t build(tinymind::QTreeNode<T>* pool, std::size_t& count, const int depth)
{
    const int16_t idx = static_cast<int16_t>(count++);
    tinymind::QTreeNode<T>& node = pool[idx];
    if (depth == MAX_DEPTH || draw(3) == 0)
    {
        node.feature = -1;
        node.threshold = 0;
        node.left = -1;
        node.right = -1;
        node.leaf_value = (draw(4) == 0) ? static_cast<int32_t>(draw(2000001)) - 1000000
                                         : static_cast<int32_t>(draw(21)) - 10;
        return idx;
    }
    node.feature = static_cast<int16_t>(draw(FEATURES));
    node.threshold = randT<T>();
    node.leaf_value = 0;
    const int16_t left = build(pool, count, depth + 1);
    const int16_t right = build(pool, count, depth + 1);
    pool[idx].left = left;
    pool[idx].right = right;
    return idx;
}

static uint32_t gHash;

static void hashI32(const int32_t v)
{
    const uint32_t u = static_cast<uint32_t>(v);
    for (int b = 0; b < 4; ++b)
    {
        gHash ^= (u >> (8 * b)) & 0xFFu;
        gHash *= 0x01000193u;
    }
}

template<typename T>
static void run(const char* name, const uint32_t seed)
{
    gState = seed;
    for (std::size_t m = 0; m < MODELS; ++m)
    {
        static tinymind::QTreeNode<T> pool[NODES];
        int16_t roots[TREES];
        int16_t classes[TREES];
        int32_t base[CLASSES];
        std::size_t count = 0;
        for (std::size_t t = 0; t < TREES; ++t)
        {
            roots[t] = build(pool, count, 0);
            classes[t] = static_cast<int16_t>(draw(CLASSES));
        }
        const bool hasBase = draw(2) == 0;
        for (std::size_t c = 0; c < CLASSES; ++c)
        {
            base[c] = static_cast<int32_t>(draw(41)) - 20;
        }
        for (std::size_t i = count; i < NODES; ++i)
        {
            pool[i].feature = -1;
            pool[i].threshold = 0;
            pool[i].left = -1;
            pool[i].right = -1;
            pool[i].leaf_value = 0;
        }

        tinymind::QGBDT<T, T, TREES, CLASSES, NODES, FEATURES> gbdt;
        gbdt.nodes = pool;
        gbdt.tree_root = roots;
        gbdt.tree_class = classes;
        gbdt.base_score = hasBase ? base : nullptr;
        tinymind::QDecisionTree<T, T, NODES, FEATURES> tree;
        tree.nodes = pool;

        gHash = 0x811C9DC5u;
        for (std::size_t s = 0; s < INPUTS; ++s)
        {
            T x[FEATURES];
            for (std::size_t f = 0; f < FEATURES; ++f)
            {
                if (draw(2) == 0)
                {
                    // Separate statements: the order of the draws is fixed.
                    const tinymind::QTreeNode<T>& n = pool[draw(static_cast<uint32_t>(count))];
                    int32_t near = (n.feature >= 0) ? static_cast<int32_t>(n.threshold)
                                                    : static_cast<int32_t>(randT<T>());
                    near += static_cast<int32_t>(draw(3)) - 1;
                    const int32_t hi = Range<T>::lo + static_cast<int32_t>(Range<T>::span) - 1;
                    x[f] = static_cast<T>(near < Range<T>::lo ? Range<T>::lo : (near > hi ? hi : near));
                }
                else
                {
                    x[f] = randT<T>();
                }
            }
            int32_t logits[CLASSES];
            gbdt.forward(x, logits);
            for (std::size_t c = 0; c < CLASSES; ++c)
            {
                hashI32(logits[c]);
            }
            hashI32(static_cast<int32_t>(gbdt.predict(x)));
            hashI32(tree.predict(x));
        }
        std::printf("%s model %2zu: nodes=%3zu base=%d hash=0x%08X\n", name, m, count,
                    hasBase ? 1 : 0, static_cast<unsigned>(gHash));
    }
}

int main()
{
    run<int8_t>("i8", 1);
    run<uint8_t>("u8", 2);
    run<int16_t>("i16", 3);
    return 0;
}
