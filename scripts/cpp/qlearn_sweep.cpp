// tinymind's QLearner (QValueTablePolicy, ArgMaxPolicy, DefaultLearningPolicy)
// on the maze example and on random environments, with the Q table hashed
// after every update.
//
// - maze: examples/maze exactly (unsigned Q16.16, glibc rand seeded with 7,
//   500 training and 100 test episodes). The Rust example runs signed Q16.16;
//   matching this hash shows the two give the same Q values here.
// - random: signed Q16.16 and Q8.8 environments of several shapes. Each state
//   has 0 to ACTIONS valid actions (so every ArgMaxPolicy branch runs: none,
//   one, two, and three or more, tied or not), random next states and
//   rewards in [-1.5, 1.5], so Q values go negative. Learning rates,
//   discount factors and exploration rates are random, including 0 and values
//   the setters must ignore (>= 1, > 100). Exploration decays during the run.
//   Every fourth episode starts with an update before any action is taken.
//
// Each line hashes (FNV-1a) every action taken and every Q value (as a 32-bit
// raw value) after every update. The draws come in a fixed order, which
// tests/qlearn_sweep.rs repeats. tests/golden/qlearn/stdout.txt is this
// program's output.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "qformat.hpp"
#include "qlearn.hpp"

typedef uint8_t state_t;
typedef uint8_t action_t;

static uint32_t gState = 0x5EED0007u;

static uint32_t draw(const uint32_t n)
{
    gState = gState * 1664525u + 1013904223u;
    return (gState >> 8) % n;
}

static uint32_t fnv(uint32_t h, const uint32_t v)
{
    for (int i = 0; i < 4; ++i)
    {
        h ^= (v >> (8 * i)) & 0xFFu;
        h *= 0x01000193u;
    }
    return h;
}

// ---- maze (examples/maze/mazelearner.h, unchanged in substance) ----

template<typename T>
struct GlibcPolicy
{
    GlibcPolicy() : mPoint(0)
    {
        srand(7U);
    }
    size_t getRandomActionDecisionPoint() const
    {
        return mPoint;
    }
    T randInt(const T min, const T max)
    {
        const T value = (rand() % (max + 1 - min)) + min;
        return value;
    }
    void setRandomActionDecisionPoint(const size_t p)
    {
        if (p <= 100)
        {
            mPoint = p;
        }
    }
private:
    size_t mPoint;
};

typedef tinymind::QValue<16, 16, false> UQ;

struct MazeEnv : public tinymind::QLearningEnvironment<state_t, action_t, UQ, 6, 6, GlibcPolicy>
{
    typedef tinymind::QLearningEnvironment<state_t, action_t, UQ, 6, 6, GlibcPolicy> ParentType;
    MazeEnv(const UQ& lr, const UQ& df, const size_t p) : ParentType(lr, df, p) {}

    static bool valid(const int s, const int a)
    {
        switch (s)
        {
            case 0: return a == 4;
            case 1: return a == 3 || a == 5;
            case 2: return a == 3;
            case 3: return a == 1 || a == 4 || a == 2;
            case 4: return a == 0 || a == 5 || a == 3;
            case 5: return a == 1 || a == 4 || a == 5;
            default: return false;
        }
    }
    static UQ reward(const int a)
    {
        return (a == 5) ? UQ(1, 0) : UQ(0, 0);
    }
    size_t getValidActionsForState(const state_t s, action_t* p) const
    {
        size_t n = 0;
        for (action_t a = 0; a < 6; ++a)
        {
            if (valid(s, a))
            {
                p[n++] = a;
            }
        }
        return n;
    }
    void takeAction(const ParentType::experience_t&) {}
};

typedef tinymind::QLearner<MazeEnv> MazeLearner;

static uint32_t hashMazeTable(uint32_t h, const MazeLearner& q)
{
    for (state_t s = 0; s < 6; ++s)
    {
        for (action_t a = 0; a < 6; ++a)
        {
            h = fnv(h, static_cast<uint32_t>(q.getQValue(s, a).getValue()));
        }
    }
    return h;
}

static void maze()
{
    MazeLearner q(UQ(1, 0) / UQ(5, 0), UQ(8, 0) / UQ(10, 0), 100);
    uint32_t h = 0x811C9DC5u;
    unsigned steps = 0;
    for (unsigned i = 0; i < 600; ++i)
    {
        if (i >= 400 && i < 500)
        {
            const size_t p = q.getEnvironment().getRandomActionDecisionPoint();
            if (p > 0)
            {
                q.getEnvironment().setRandomActionDecisionPoint(p - 1);
            }
        }
        q.startNewEpisode();
        state_t state = rand() % 6;
        do
        {
            const action_t action = q.takeAction(state);
            MazeEnv::experience_t e;
            e.state = state;
            e.action = action;
            e.reward = MazeEnv::reward(action);
            e.newState = action;
            q.updateFromExperience(e);
            h = fnv(h, action);
            h = hashMazeTable(h, q);
            ++steps;
            state = action;
        } while (q.getState() != 5);
    }
    std::printf("maze steps=%u hash=0x%08X q=", steps, h);
    for (state_t s = 0; s < 6; ++s)
    {
        for (action_t a = 0; a < 6; ++a)
        {
            std::printf("%s%u", (s || a) ? "," : "", static_cast<unsigned>(q.getQValue(s, a).getValue()));
        }
    }
    std::printf("\n");
}

// ---- random environments ----

template<typename T>
struct LcgPolicy
{
    LcgPolicy() : mPoint(0) {}
    size_t getRandomActionDecisionPoint() const
    {
        return mPoint;
    }
    T randInt(const T min, const T max)
    {
        return static_cast<T>(draw(static_cast<uint32_t>(max + 1 - min)) + min);
    }
    void setRandomActionDecisionPoint(const size_t p)
    {
        if (p <= 100)
        {
            mPoint = p;
        }
    }
private:
    size_t mPoint;
};

template<typename V, size_t S, size_t A>
struct RandomEnv : public tinymind::QLearningEnvironment<state_t, action_t, V, S, A, LcgPolicy>
{
    typedef tinymind::QLearningEnvironment<state_t, action_t, V, S, A, LcgPolicy> ParentType;
    RandomEnv(const V& lr, const V& df, const size_t p) : ParentType(lr, df, p) {}

    size_t getValidActionsForState(const state_t s, action_t* p) const
    {
        size_t n = 0;
        for (action_t a = 0; a < A; ++a)
        {
            if (s < S && valid[s][a])
            {
                p[n++] = a;
            }
        }
        return n;
    }
    void takeAction(const typename ParentType::experience_t&) {}

    bool valid[S][A];
    state_t next[S][A];
    V reward[S][A];
};

// Raw value of 1.0 is 2^FRAC; draws a raw value for a learning rate or
// discount: 0, exactly 1, above 1 (both ignored by the setter), or in (0, 1).
static int32_t rateRaw(const int frac)
{
    const int32_t one = 1 << frac;
    switch (draw(7))
    {
        case 0: return 0;
        case 1: return one;
        case 2: return one + 1 + static_cast<int32_t>(draw(static_cast<uint32_t>(one)));
        default: return 1 + static_cast<int32_t>(draw(static_cast<uint32_t>(one - 1)));
    }
}

template<typename V, int FRAC, size_t S, size_t A>
static void randomEnv(const char* name, const int trials)
{
    typedef RandomEnv<V, S, A> Env;
    typedef tinymind::QLearner<Env> Learner;
    for (int t = 0; t < trials; ++t)
    {
        const int32_t lr = rateRaw(FRAC);
        const int32_t df = rateRaw(FRAC);
        const uint32_t pk = draw(4);
        const size_t point = (pk == 0) ? 0 : (pk == 1) ? 100 : (pk == 2) ? 150 : 1 + draw(99);
        Learner q(V(lr), V(df), point);
        Env& env = q.getEnvironment();
        for (size_t s = 0; s < S; ++s)
        {
            // 0, 1, 2 or any number of valid actions.
            const uint32_t kind = draw(5);
            for (size_t a = 0; a < A; ++a)
            {
                bool v;
                if (kind == 0)
                {
                    v = false;
                }
                else if (kind == 1)
                {
                    v = (a == 0);
                }
                else if (kind == 2)
                {
                    v = (a == 0) || (a + 1 == A);
                }
                else
                {
                    v = draw(3) != 0;
                }
                env.valid[s][a] = v;
                env.next[s][a] = static_cast<state_t>(draw(S));
                const int32_t span = 3 << (FRAC - 1); // 1.5
                env.reward[s][a] = V(static_cast<int32_t>(draw(2 * span + 1)) - span);
            }
        }
        const state_t goal = static_cast<state_t>(S - 1);
        uint32_t h = 0x811C9DC5u;
        unsigned steps = 0;
        for (int ep = 0; ep < 60; ++ep)
        {
            if (ep >= 30)
            {
                const size_t p = env.getRandomActionDecisionPoint();
                if (p > 0)
                {
                    env.setRandomActionDecisionPoint(p - 1);
                }
            }
            q.startNewEpisode();
            state_t state = static_cast<state_t>(draw(S));
            if (ep % 4 == 3)
            {
                // An update with no current state: it only moves the learner.
                typename Env::experience_t e;
                e.state = state;
                e.action = static_cast<action_t>(draw(A));
                e.reward = env.reward[state][e.action];
                e.newState = env.next[state][e.action];
                q.updateFromExperience(e);
                h = fnv(h, q.getState());
                state = e.newState;
            }
            for (int step = 0; step < 25; ++step)
            {
                const action_t action = q.takeAction(state);
                h = fnv(h, action);
                if (action == Learner::InvalidAction)
                {
                    break;
                }
                typename Env::experience_t e;
                e.state = state;
                e.action = action;
                e.reward = env.reward[state][action];
                e.newState = env.next[state][action];
                q.updateFromExperience(e);
                for (state_t s = 0; s < S; ++s)
                {
                    for (action_t a = 0; a < A; ++a)
                    {
                        h = fnv(h, static_cast<uint32_t>(static_cast<int32_t>(q.getQValue(s, a).getValue())));
                    }
                }
                ++steps;
                state = e.newState;
                if (state == goal)
                {
                    break;
                }
            }
        }
        std::printf("%s %d lr=%d df=%d point=%u steps=%u hash=0x%08X\n", name, t, lr, df,
                    static_cast<unsigned>(point), steps, h);
    }
}

int main()
{
    maze();
    randomEnv<tinymind::QValue<16, 16, true>, 16, 6, 6>("q16.16_6x6", 12);
    randomEnv<tinymind::QValue<16, 16, true>, 16, 4, 2>("q16.16_4x2", 12);
    randomEnv<tinymind::QValue<16, 16, true>, 16, 10, 5>("q16.16_10x5", 12);
    randomEnv<tinymind::QValue<8, 8, true>, 8, 6, 6>("q8.8_6x6", 12);
    randomEnv<tinymind::QValue<8, 8, true>, 8, 3, 3>("q8.8_3x3", 12);
    return 0;
}
