//! Minimal OpenQASM 2.0 parser.
//!
//! Parses a useful subset of QASM into a [`Circuit`](crate::circuit::Circuit):
//!
//! ```qasm
//! OPENQASM 2.0;
//! include "qelib1.inc";
//! qreg q[2];
//! h q[0];
//! cx q[0], q[1];
//! measure q[0] -> c[0];
//! ```

use crate::circuit::Circuit;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{char, multispace0, multispace1, u64 as nom_u64},
    combinator::{map, value},
    sequence::{preceded, tuple},
    IResult,
};

/// A parsed QASM statement.
#[derive(Debug, Clone)]
enum Stmt {
    Header,
    Include,
    QReg {
        _name: String,
        size: usize,
    },
    CReg {
        _name: String,
        _size: usize,
    },
    Gate1 {
        name: String,
        target: usize,
    },
    Gate2 {
        name: String,
        control: usize,
        target: usize,
    },
    Measure {
        qubit: usize,
        _bit: usize,
    },
}

// ── Helpers ─────────────────────────────────────────────────────────

fn ws(input: &str) -> IResult<&str, ()> {
    value((), multispace0)(input)
}

fn ident(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_')(input)
}

fn qubit_ref(input: &str) -> IResult<&str, (&str, usize)> {
    let (input, name) = ident(input)?;
    let (input, _) = char('[')(input)?;
    let (input, idx) = nom_u64(input)?;
    let (input, _) = char(']')(input)?;
    Ok((input, (name, idx as usize)))
}

// ── Statement parsers ───────────────────────────────────────────────

fn header(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("OPENQASM")(input)?;
    let (input, _) = take_until(";")(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Stmt::Header))
}

fn include(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("include")(input)?;
    let (input, _) = take_until(";")(input)?;
    let (input, _) = char(';')(input)?;
    Ok((input, Stmt::Include))
}

fn qreg(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("qreg")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (name, size)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(';'))(input)?;
    Ok((
        input,
        Stmt::QReg {
            _name: name.to_string(),
            size,
        },
    ))
}

fn creg(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("creg")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (name, size)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(';'))(input)?;
    Ok((
        input,
        Stmt::CReg {
            _name: name.to_string(),
            _size: size,
        },
    ))
}

fn gate1(input: &str) -> IResult<&str, Stmt> {
    let (input, name) = alt((tag("h"), tag("x"), tag("y"), tag("z"), tag("s"), tag("t")))(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (_, target)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(';'))(input)?;
    Ok((
        input,
        Stmt::Gate1 {
            name: name.to_string(),
            target,
        },
    ))
}

fn gate2(input: &str) -> IResult<&str, Stmt> {
    let (input, name) = alt((tag("cx"), tag("cz"), tag("swap")))(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (_, control)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(','))(input)?;
    let (input, _) = ws(input)?;
    let (input, (_, target)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(';'))(input)?;
    Ok((
        input,
        Stmt::Gate2 {
            name: name.to_string(),
            control,
            target,
        },
    ))
}

fn measure(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("measure")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (_, qubit)) = qubit_ref(input)?;
    let (input, _) = tuple((ws, tag("->"), ws))(input)?;
    let (input, (_, bit)) = qubit_ref(input)?;
    let (input, _) = preceded(ws, char(';'))(input)?;
    Ok((input, Stmt::Measure { qubit, _bit: bit }))
}

fn comment(input: &str) -> IResult<&str, ()> {
    let (input, _) = tag("//")(input)?;
    let (input, _) = take_until("\n")(input)?;
    Ok((input, ()))
}

fn statement(input: &str) -> IResult<&str, Option<Stmt>> {
    let (input, _) = ws(input)?;
    alt((
        map(comment, |_| None),
        map(header, Some),
        map(include, Some),
        map(qreg, Some),
        map(creg, Some),
        map(gate2, Some), // try two-qubit before one-qubit (cx vs c*)
        map(gate1, Some),
        map(measure, Some),
    ))(input)
}

/// Parse an OpenQASM 2.0 string into a [`Circuit`].
pub fn parse(source: &str) -> Result<Circuit, String> {
    let mut input = source;
    let mut stmts = Vec::new();

    while !input.trim().is_empty() {
        match statement(input) {
            Ok((rest, maybe_stmt)) => {
                if let Some(s) = maybe_stmt {
                    stmts.push(s);
                }
                input = rest;
            }
            Err(e) => {
                let ctx: String = input.chars().take(40).collect();
                return Err(format!("parse error near '{ctx}': {e}"));
            }
        }
    }

    // Find qreg to determine qubit count.
    let num_qubits = stmts
        .iter()
        .filter_map(|s| match *s {
            Stmt::QReg { size, .. } => Some(size),
            _ => None,
        })
        .sum();

    if num_qubits == 0 {
        return Err("no qreg declaration found".into());
    }

    let mut circuit = Circuit::new(num_qubits);

    for stmt in &stmts {
        match *stmt {
            Stmt::Gate1 { ref name, target } => match name.as_str() {
                "h" => {
                    circuit.h(target);
                }
                "x" => {
                    circuit.x(target);
                }
                "y" => {
                    circuit.y(target);
                }
                "z" => {
                    circuit.z(target);
                }
                "s" => {
                    circuit.s(target);
                }
                "t" => {
                    circuit.t(target);
                }
                other => return Err(format!("unknown gate: {other}")),
            },
            Stmt::Gate2 {
                ref name,
                control,
                target,
            } => match name.as_str() {
                "cx" => {
                    circuit.cnot(control, target);
                }
                "cz" => {
                    circuit.cz(control, target);
                }
                "swap" => {
                    circuit.swap(control, target);
                }
                other => return Err(format!("unknown gate: {other}")),
            },
            Stmt::Measure { qubit, .. } => {
                circuit.add_measure(qubit);
            }
            _ => {}
        }
    }

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bell_circuit() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let circuit = parse(qasm).expect("should parse");
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.ops.len(), 4); // h, cx, measure, measure
    }

    #[test]
    fn roundtrip_bell_state() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n";
        let circuit = parse(qasm).unwrap();
        let result = circuit.run();
        let probs = result.state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
}
