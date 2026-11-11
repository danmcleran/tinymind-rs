// The Q-learning sweep, shared by tests/qlearn_sweep.rs (on the host,
// against tinymind (C++)'s output) and bench/src/sweeps.rs (on Cortex-M,
// against the same golden). no_std and allocation-free.

use core::fmt::Write;

use tinymind::qlearn::{Environment, Experience, QLearner};
use tinymind::{Q8_8, Q16_16, Scalar};

pub fn fnv(mut h: u32, v: u32) -> u32 {
    for b in v.to_le_bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

#[derive(Clone, Copy)]
struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }
}

trait Format: Scalar {
    const FRAC: u32;
    fn from_raw32(raw: i32) -> Self;
    fn raw32(self) -> u32;
}

impl Format for Q16_16 {
    const FRAC: u32 = 16;
    fn from_raw32(raw: i32) -> Self {
        Self::from_raw(raw)
    }
    fn raw32(self) -> u32 {
        self.raw() as u32
    }
}

impl Format for Q8_8 {
    const FRAC: u32 = 8;
    fn from_raw32(raw: i32) -> Self {
        Self::from_raw(raw as i16)
    }
    fn raw32(self) -> u32 {
        i32::from(self.raw()) as u32
    }
}

struct RandomEnv<V, const S: usize, const A: usize> {
    lcg: Lcg,
    valid: [[bool; A]; S],
    next: [[usize; A]; S],
    reward: [[V; A]; S],
}

impl<V, const S: usize, const A: usize> Environment<A> for RandomEnv<V, S, A> {
    fn valid_actions(&self, state: usize, actions: &mut [usize; A]) -> usize {
        let mut n = 0;
        if let Some(row) = self.valid.get(state) {
            for (a, &v) in row.iter().enumerate() {
                if v {
                    actions[n] = a;
                    n += 1;
                }
            }
        }
        n
    }

    fn random_int(&mut self, min: usize, max: usize) -> usize {
        self.lcg.draw((max + 1 - min) as u32) as usize + min
    }
}

fn rate_raw(r: &mut Lcg, frac: u32) -> i32 {
    let one = 1i32 << frac;
    match r.draw(8) {
        0 => 0,
        1 => one,
        2 => one + 1 + r.draw(one as u32) as i32,
        3 => -1 - r.draw(one as u32) as i32,
        _ => 1 + r.draw((one - 1) as u32) as i32,
    }
}

fn hash_table<
    V: Format,
    Q: tinymind::qlearn::QValues<V>,
    E: Environment<A>,
    const S: usize,
    const A: usize,
>(
    mut h: u32,
    q: &QLearner<V, E, S, A, Q>,
) -> u32 {
    for s in 0..S {
        for a in 0..A {
            h = fnv(h, q.q_value(s, a).raw32());
        }
    }
    h
}

fn random_env<V: Format, const S: usize, const A: usize>(
    lcg: &mut Lcg,
    name: &str,
    trials: usize,
    out: &mut impl Write,
) {
    for t in 0..trials {
        let lr = rate_raw(lcg, V::FRAC);
        let df = rate_raw(lcg, V::FRAC);
        let point = match lcg.draw(4) {
            0 => 0,
            1 => 100,
            2 => 150,
            _ => 1 + lcg.draw(99) as usize,
        };
        let mut env = RandomEnv::<V, S, A> {
            lcg: *lcg,
            valid: [[false; A]; S],
            next: [[0; A]; S],
            reward: [[V::ZERO; A]; S],
        };
        for s in 0..S {
            let kind = env.lcg.draw(5);
            for a in 0..A {
                env.valid[s][a] = match kind {
                    0 => false,
                    1 => a == 0,
                    2 => a == 0 || a + 1 == A,
                    _ => env.lcg.draw(3) != 0,
                };
                env.next[s][a] = env.lcg.draw(S as u32) as usize;
                let span = 3i32 << (V::FRAC - 1);
                env.reward[s][a] = V::from_raw32(env.lcg.draw((2 * span + 1) as u32) as i32 - span);
            }
        }
        let mut q = QLearner::<V, _, S, A>::new(env, V::from_raw32(lr), V::from_raw32(df), point);
        let goal = S - 1;
        let mut h = 0x811C_9DC5u32;
        let mut steps = 0u32;
        for ep in 0..60 {
            if ep >= 30 {
                let p = q.random_action_decision_point();
                if p > 0 {
                    q.set_random_action_decision_point(p - 1);
                }
            }
            q.start_new_episode();
            let mut state = q.environment_mut().lcg.draw(S as u32) as usize;
            if ep % 4 == 3 {
                // An update before any action, for a random state or the
                // invalid state S.
                let env = q.environment_mut();
                let es = env.lcg.draw(S as u32 + 1) as usize;
                let action = env.lcg.draw(A as u32) as usize;
                let reward = if es < S {
                    env.reward[es][action]
                } else {
                    V::ZERO
                };
                let new_state = env.next[es % S][action];
                q.update_from_experience(&Experience {
                    state: es,
                    action,
                    reward,
                    new_state,
                });
                h = fnv(h, q.state().unwrap() as u32);
                h = hash_table(h, &q);
                state = new_state;
            }
            for step in 0..25 {
                let Some(action) = q.take_action(state) else {
                    h = fnv(h, 255); // the C++ InvalidAction
                    break;
                };
                h = fnv(h, action as u32);
                let env = q.environment();
                let (reward, new_state) = (env.reward[state][action], env.next[state][action]);
                q.update_from_experience(&Experience {
                    state,
                    action,
                    reward,
                    new_state,
                });
                h = hash_table(h, &q);
                steps += 1;
                state = new_state;
                if step % 5 == 4 {
                    // A replayed experience for a random state.
                    let env = q.environment_mut();
                    let rs = env.lcg.draw(S as u32) as usize;
                    let ra = env.lcg.draw(A as u32) as usize;
                    let (reward, new_state) = (env.reward[rs][ra], env.next[rs][ra]);
                    q.update_from_experience(&Experience {
                        state: rs,
                        action: ra,
                        reward,
                        new_state,
                    });
                    h = hash_table(h, &q);
                    state = new_state;
                }
                if state == goal {
                    break;
                }
            }
        }
        *lcg = q.environment().lcg;
        writeln!(
            out,
            "{name} {t} lr={lr} df={df} point={point} steps={steps} hash=0x{h:08X}"
        )
        .unwrap();
    }
}

/// Write the random environments' part of the report
/// (tests/golden/qlearn/stdout.txt after its first line, the maze, which
/// tests/qlearn_sweep.rs writes: the maze needs
/// examples/common/maze_driver.rs, which uses std).
pub fn report(out: &mut impl Write) {
    let mut lcg = Lcg(0x5EED_0007);
    random_env::<Q16_16, 6, 6>(&mut lcg, "q16.16_6x6", 12, out);
    random_env::<Q16_16, 4, 2>(&mut lcg, "q16.16_4x2", 12, out);
    random_env::<Q16_16, 10, 5>(&mut lcg, "q16.16_10x5", 12, out);
    random_env::<Q8_8, 6, 6>(&mut lcg, "q8.8_6x6", 12, out);
    random_env::<Q8_8, 3, 3>(&mut lcg, "q8.8_3x3", 12, out);
}
