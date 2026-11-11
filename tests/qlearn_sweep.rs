// QLearner against tinymind's QLearner on the maze and on random
// environments. tests/golden/qlearn/stdout.txt is the output of
// scripts/cpp/qlearn_sweep.cpp, which runs the same episodes from the same
// generators; scripts/compare-with-cpp.sh rebuilds it and checks the golden is
// current. See that program for what is covered.
// The random environments are in tests/sweeps/qlearn.rs, which bench/ also
// runs on Cortex-M; the maze stays here.

#[allow(dead_code)]
#[path = "../examples/common/maze_driver.rs"]
mod maze_driver;
#[path = "sweeps/qlearn.rs"]
mod qlearn;

use std::fmt::Write;

use qlearn::fnv;
use tinymind::Q16_16;
use tinymind::qlearn::Experience;

fn maze(out: &mut String) {
    let mut q = maze_driver::new_learner();
    let mut h = 0x811C_9DC5u32;
    let mut steps = 0u32;
    for i in 0..5100 {
        if (4900..5000).contains(&i) {
            let p = q.random_action_decision_point();
            if p > 0 {
                q.set_random_action_decision_point(p - 1);
            }
        }
        if i == 5000 {
            q.set_random_action_decision_point(0);
            q.set_learning_rate(Q16_16::ZERO);
        }
        q.start_new_episode();
        let mut state = (q.environment_mut().rng.next_i32() % 6) as usize;
        loop {
            let action = q.take_action(state).unwrap();
            let reward = q.environment().reward(state, action);
            q.update_from_experience(&Experience {
                state,
                action,
                reward,
                new_state: action,
            });
            h = fnv(h, action as u32);
            for s in 0..6 {
                for a in 0..6 {
                    h = fnv(h, q.q_value(s, a).raw() as u32);
                }
            }
            steps += 1;
            state = action;
            if q.state() == Some(5) {
                break;
            }
        }
    }
    let mut table = Vec::new();
    for s in 0..6 {
        for a in 0..6 {
            table.push((q.q_value(s, a).raw() as u32).to_string());
        }
    }
    writeln!(
        out,
        "maze steps={steps} hash=0x{h:08X} q={}",
        table.join(",")
    )
    .unwrap();
}

#[test]
fn qlearn_matches_cpp_golden() {
    let mut out = String::new();
    maze(&mut out);
    qlearn::report(&mut out);
    let golden = include_str!("golden/qlearn/stdout.txt");
    for (i, (a, g)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, g, "line {}", i + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count(), "line count");
}
