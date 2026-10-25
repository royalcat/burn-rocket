//! Dump planner and regcmd output for the C oracle comparison.
//!
//! Usage:
//!   regcmd_dump plan         M K N
//!   regcmd_dump plan-int8    M K N
//!   regcmd_dump plan-int8-gw M K N [GROUP]
//!   regcmd_dump plan-bf16    M K N
//!   regcmd_dump emit         M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-int8    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-bf16    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!
//! The emitters take the TILE dims (Mt Kt Nt), not the full problem dims, and a
//! plan prints `Mt Kt Nt jobs` or `-1`, matching `tools/c-oracle/oracle`.
//!
//! int4 and tf32 have no Rust planner/emitter yet (M3/M4). Their subcommands
//! exist only so the parity harness can report them as PENDING; they exit
//! nonzero without writing a plan/program.
//!
//! `emit-int8` with ACCUMULATE=1 is not ported: the C int8 emitter has an
//! EW-add path there, but the C library itself documents that encoding as
//! non-bit-exact and never used in production, so the Rust port leaves it out.

use std::process::ExitCode;

use rocket_npu::matmul::plan::{self, BF16, INT8, INT8_GW, Plan};
use rocket_npu::regcmd::fp16::{MatmulParams, emit};

fn dma(arg: &str) -> u32 {
    let (radix, digits) = match arg.strip_prefix("0x") {
        Some(d) => (16, d),
        None => (10, arg),
    };
    u32::from_str_radix(digits, radix).unwrap_or(0)
}

fn parse_dims(args: &[String]) -> Option<(usize, usize, usize)> {
    Some((
        args.get(2)?.parse().ok()?,
        args.get(3)?.parse().ok()?,
        args.get(4)?.parse().ok()?,
    ))
}

fn print_plan(geometry: &plan::Geometry, group: usize, m: usize, k: usize, n: usize) {
    match plan::plan_geom(geometry, m, k, n, group) {
        Ok((mt, kt, nt)) => {
            let jobs = m.div_ceil(mt) * n.div_ceil(nt) * k.div_ceil(kt);
            println!("{mt} {kt} {nt} {jobs}");
        }
        Err(e) => {
            println!("-1");
            eprintln!("{e}");
        }
    }
}

fn emit_program(kind: &str, params: &MatmulParams) -> ExitCode {
    let mut ops = Vec::new();
    let r = match kind {
        "emit" => emit(params, &mut ops),
        "emit-int8" => rocket_npu::regcmd::int8::emit(params, &mut ops),
        "emit-bf16" => rocket_npu::regcmd::bf16::emit(params, &mut ops),
        _ => unreachable!(),
    };
    match r {
        Ok(count) => {
            println!("{count}");
            for op in ops {
                println!("{:016x}", op.0);
            }
        }
        Err(e) => {
            println!("-1");
            eprintln!("{e}");
        }
    }
    ExitCode::SUCCESS
}

fn not_implemented(mode: &str, milestone: &str) -> ExitCode {
    eprintln!("{mode}: not implemented in the Rust port yet ({milestone})");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: regcmd_dump plan|plan-int8|plan-int8-gw|plan-bf16 M K N [GROUP]");
        eprintln!(
            "       regcmd_dump emit|emit-int8|emit-bf16 M K N IN WT OUT [FP32TOFP16] [ACCUMULATE] [ADD_DMA]"
        );
        return ExitCode::FAILURE;
    }
    let Some((m, k, n)) = parse_dims(&args) else {
        eprintln!("bad dimensions");
        return ExitCode::FAILURE;
    };
    let mode = args[1].as_str();

    match mode {
        "plan" => {
            match Plan::new(m, k, n) {
                Ok(p) => println!("{} {} {} {}", p.mt, p.kt, p.nt, p.jobs()),
                Err(e) => {
                    println!("-1");
                    eprintln!("{e}");
                }
            }
            ExitCode::SUCCESS
        }
        "plan-int8" => {
            print_plan(&INT8, 0, m, k, n);
            ExitCode::SUCCESS
        }
        "plan-int8-gw" => {
            let group: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(0);
            print_plan(&INT8_GW, group, m, k, n);
            ExitCode::SUCCESS
        }
        "plan-bf16" => {
            print_plan(&BF16, 0, m, k, n);
            ExitCode::SUCCESS
        }
        "plan-int4" => not_implemented(mode, "M3"),
        "plan-tf32" => not_implemented(mode, "M4"),
        "emit" | "emit-int8" | "emit-bf16" => {
            if args.len() < 8 {
                eprintln!("emit needs M K N IN WT OUT");
                return ExitCode::FAILURE;
            }
            let mut p = MatmulParams::new(
                m as u32,
                k as u32,
                n as u32,
                dma(&args[5]),
                dma(&args[6]),
                dma(&args[7]),
            );
            if let Some(v) = args.get(8) {
                p.fp32_to_fp16 = v != "0";
            }
            if let Some(v) = args.get(9) {
                p.accumulate = v != "0";
            }
            if let Some(v) = args.get(10) {
                p.add_dma = dma(v);
            }
            emit_program(mode, &p)
        }
        "emit-int4" => not_implemented(mode, "M3"),
        "emit-tf32" => not_implemented(mode, "M4"),
        other => {
            eprintln!("unknown mode {other}");
            ExitCode::FAILURE
        }
    }
}
