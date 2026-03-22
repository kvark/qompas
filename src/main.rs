//! Qompas — Quantum Exploration Tool
//!
//! Entry point that demonstrates building and running a small circuit.

use qompas::circuit::Circuit;
use qompas::formats::openqasm;

fn main() {
    env_logger::init();

    println!("=== Qompas — Quantum Exploration Library ===\n");

    // Build a Bell state circuit programmatically.
    let mut bell = Circuit::new(2);
    bell.h(0).cnot(0, 1);

    let result = bell.run();
    let probs = result.state.probabilities();
    println!("Bell state (programmatic):");
    print_probs(&probs, 2);

    // Parse the same circuit from OpenQASM.
    let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n";
    let circuit = openqasm::parse(qasm).expect("valid QASM");
    let result = circuit.run();
    let probs = result.state.probabilities();
    println!("\nBell state (from OpenQASM):");
    print_probs(&probs, 2);

    // 3-qubit GHZ state.
    let mut ghz = Circuit::new(3);
    ghz.h(0).cnot(0, 1).cnot(0, 2);
    let result = ghz.run();
    let probs = result.state.probabilities();
    println!("\nGHZ state:");
    print_probs(&probs, 3);

    println!("\nDone. Future versions will launch the blade-graphics + egui visualizer.");
}

fn print_probs(probs: &[f64], n: usize) {
    for (i, p) in probs.iter().enumerate() {
        if *p > 1e-10 {
            println!("  |{:0>width$b}⟩  {:.4}", i, p, width = n);
        }
    }
}
