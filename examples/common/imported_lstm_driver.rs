//! The imported_lstm example's program, shared by `examples/imported_lstm.rs`
//! and `tests/imported_lstm_golden.rs` (which runs it in-process against the
//! C++ output).

use std::io::{self, Write};

use tinymind::int8::calibration::quantize;
use tinymind::int8::{CellStorage, QLstmCell, Requantizer};

#[path = "imported_lstm_model.rs"]
mod lstm;

#[path = "imported_lstm_i8_model.rs"]
mod lstm_i8;

/// What the C++ check prints, model by model: the LSTM cell's multipliers and
/// shifts, hashes of its biases and tables, each dense layer's requantizer and
/// the softmax parameters or a hash of the tanh table; then for 16 test
/// sequences a hash of the hidden and cell state after every step and the
/// outputs.
pub fn run(out: &mut dyn Write) -> io::Result<()> {
    run_lstm(out)?;
    run_lstm_i8(out)
}

/// FNV-1a over bytes, continuing from `h`.
fn fnv(h: u32, bytes: impl IntoIterator<Item = u8>) -> u32 {
    bytes
        .into_iter()
        .fold(h, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

fn pairs(m: &[i32; 4], s: &[i32; 4]) -> String {
    let v: Vec<String> = m.iter().zip(s).map(|(m, s)| format!("{m},{s}")).collect();
    v.join(";")
}

/// The cell's integer parameters, as the C++ prints the ones it builds.
fn print_cell<const IN: usize, const HID: usize, C: CellStorage>(
    out: &mut dyn Write,
    model: &str,
    cell: &QLstmCell<'_, IN, HID, C>,
) -> io::Result<()> {
    writeln!(
        out,
        "{model} lstm in={} rec={} fc={},{} ig={},{} ct={},{} out={},{} cell={},{}",
        pairs(&cell.input_to_lut_multiplier, &cell.input_to_lut_shift),
        pairs(
            &cell.recurrent_to_lut_multiplier,
            &cell.recurrent_to_lut_shift
        ),
        cell.f_times_c_multiplier,
        cell.f_times_c_shift,
        cell.i_times_g_multiplier,
        cell.i_times_g_shift,
        cell.cell_to_tanh_multiplier,
        cell.cell_to_tanh_shift,
        cell.output_requantizer.multiplier,
        cell.output_requantizer.shift,
        cell.cell_qmin.into(),
        cell.cell_qmax.into(),
    )?;
    let biases = cell.biases.iter().flatten().flat_map(|b| b.to_le_bytes());
    let luts = cell
        .sigmoid_lut
        .iter()
        .chain(cell.tanh_lut)
        .chain(cell.tanh_cell_lut)
        .map(|&v| v as u8);
    writeln!(
        out,
        "{model} lstm biases=0x{:08X} luts=0x{:08X}",
        fnv(0x811C_9DC5, biases),
        fnv(0x811C_9DC5, luts)
    )
}

fn print_rq(out: &mut dyn Write, model: &str, name: &str, rq: &Requantizer) -> io::Result<()> {
    writeln!(
        out,
        "{model} {name} rq={},{},{}",
        rq.multiplier, rq.shift, rq.zero_point
    )
}

/// Test sequence `s`, `[step][feature]`, as the C++ computes it.
fn input<const T: usize, const F: usize>(s: usize, scale: f32, zero_point: i8) -> [[i8; F]; T] {
    core::array::from_fn(|t| {
        core::array::from_fn(|k| {
            let real = ((s * 23 + t * 29 + k * 13 + s * t * 7) % 37) as f32 / 18.0 - 1.0;
            quantize(real, scale, i32::from(zero_point), -128, 127)
        })
    })
}

fn joined(v: &[i8]) -> String {
    let v: Vec<String> = v.iter().map(i8::to_string).collect();
    v.join(",")
}

fn run_lstm(out: &mut dyn Write) -> io::Result<()> {
    print_cell(out, "lstm", &lstm::LSTM1)?;
    print_rq(out, "lstm", "fc1", &lstm::FC1.requantizer)?;
    let p = lstm::PROBS;
    writeln!(
        out,
        "lstm softmax={},{},{}",
        p.input_multiplier, p.input_left_shift, p.diff_min
    )?;
    for s in 0..16 {
        let x: lstm::Input = input(s, lstm::INPUT_SCALE, lstm::INPUT_ZERO_POINT);
        let mut hidden = [lstm::LSTM1_ZERO_POINT; lstm::LSTM1_HIDDEN];
        let mut cell = [0i16; lstm::LSTM1_HIDDEN];
        let mut h = 0x811C_9DC5;
        for step in &x {
            lstm::LSTM1.forward(step, &mut hidden, &mut cell);
            h = fnv(h, hidden.iter().map(|&v| v as u8));
            h = fnv(h, cell.iter().flat_map(|c| c.to_le_bytes()));
        }
        let probs = lstm::head(&hidden);
        assert_eq!(lstm::forward(&x), probs, "forward differs from stepping");
        writeln!(out, "lstm s={s} states=0x{h:08X} probs={}", joined(&probs))?;
    }
    Ok(())
}

fn run_lstm_i8(out: &mut dyn Write) -> io::Result<()> {
    print_cell(out, "lstm_i8", &lstm_i8::LSTM1)?;
    print_rq(out, "lstm_i8", "fc1", &lstm_i8::FC1.requantizer)?;
    print_rq(out, "lstm_i8", "fc2", &lstm_i8::FC2.requantizer)?;
    let lut = fnv(0x811C_9DC5, lstm_i8::OUT_LUT.iter().map(|&v| v as u8));
    writeln!(out, "lstm_i8 tanh lut=0x{lut:08X}")?;
    for s in 0..16 {
        let x: lstm_i8::Input = input(s, lstm_i8::INPUT_SCALE, lstm_i8::INPUT_ZERO_POINT);
        let mut hidden = [lstm_i8::LSTM1_ZERO_POINT; lstm_i8::LSTM1_HIDDEN];
        let mut cell = [0i8; lstm_i8::LSTM1_HIDDEN];
        let mut h = 0x811C_9DC5;
        for step in &x {
            lstm_i8::LSTM1.forward(step, &mut hidden, &mut cell);
            h = fnv(h, hidden.iter().map(|&v| v as u8));
            h = fnv(h, cell.iter().map(|&v| v as u8));
        }
        let y = lstm_i8::head(&hidden);
        assert_eq!(lstm_i8::forward(&x), y, "forward differs from stepping");
        writeln!(out, "lstm_i8 s={s} states=0x{h:08X} out={}", joined(&y))?;
    }
    Ok(())
}
