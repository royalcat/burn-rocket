//! Dump planner and regcmd output for the C oracle comparison.
//!
//! Usage:
//!   regcmd_dump plan M K N
//!   regcmd_dump emit M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]

use std::process::ExitCode;

use rocket_npu::matmul::plan::Plan;
use rocket_npu::regcmd::fp16::{MatmulParams, emit};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: regcmd_dump plan|emit M K N ...");
        return ExitCode::FAILURE;
    }
    let nums: Vec<usize> = match args[2..5].iter().map(|s| s.parse()).collect() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("bad dimensions");
            return ExitCode::FAILURE;
        }
    };
    match args[1].as_str() {
        "plan" => match Plan::new(nums[0], nums[1], nums[2]) {
            Ok(p) => println!("{} {} {} {}", p.mt, p.kt, p.nt, p.nmt * p.nnt * p.nkt),
            Err(e) => {
                println!("-1");
                eprintln!("{e}");
            }
        },
        "plan-int8" => {
            let group: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(0);
            match rocket_npu::matmul::plan::plan_geom(
                &rocket_npu::matmul::plan::INT8_GW,
                nums[0],
                nums[1],
                nums[2],
                group,
            ) {
                Ok((mt, kt, nt)) => {
                    let jobs = nums[0].div_ceil(mt) * nums[2].div_ceil(nt) * nums[1].div_ceil(kt);
                    println!("{mt} {kt} {nt} {jobs}");
                }
                Err(_) => println!("-1"),
            }
        }
        "plan-bf16" => match rocket_npu::matmul::plan::plan_geom(
            &rocket_npu::matmul::plan::BF16,
            nums[0],
            nums[1],
            nums[2],
            0,
        ) {
            Ok((mt, kt, nt)) => {
                let jobs = nums[0].div_ceil(mt) * nums[2].div_ceil(nt) * nums[1].div_ceil(kt);
                println!("{mt} {kt} {nt} {jobs}");
            }
            Err(_) => println!("-1"),
        },
        "emit-int8" | "emit-bf16" => {
            if args.len() < 8 {
                eprintln!("emit needs M K N IN WT OUT");
                return ExitCode::FAILURE;
            }
            let dma = |i: usize| -> u32 {
                let v = &args[i];
                let (radix, digits) = match v.strip_prefix("0x") {
                    Some(d) => (16, d),
                    None => (10, v.as_str()),
                };
                u32::from_str_radix(digits, radix).unwrap_or(0)
            };
            let p = MatmulParams::new(
                nums[0] as u32,
                nums[1] as u32,
                nums[2] as u32,
                dma(5),
                dma(6),
                dma(7),
            );
            let mut ops = Vec::new();
            let r = if args[1] == "emit-int8" {
                rocket_npu::regcmd::int8::emit(&p, &mut ops)
            } else {
                rocket_npu::regcmd::bf16::emit(&p, &mut ops)
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
        }
        "emit" => {
            if args.len() < 8 {
                eprintln!("emit needs M K N IN WT OUT");
                return ExitCode::FAILURE;
            }
            let dma = |i: usize| -> u32 {
                let v = &args[i];
                let (radix, digits) = match v.strip_prefix("0x") {
                    Some(d) => (16, d),
                    None => (10, v.as_str()),
                };
                u32::from_str_radix(digits, radix).unwrap_or(0)
            };
            let mut p = MatmulParams::new(
                nums[0] as u32,
                nums[1] as u32,
                nums[2] as u32,
                dma(5),
                dma(6),
                dma(7),
            );
            if let Some(v) = args.get(8) {
                p.fp32_to_fp16 = v != "0";
            }
            if let Some(v) = args.get(9) {
                p.accumulate = v != "0";
            }
            if let Some(v) = args.get(10) {
                let (radix, digits) = match v.strip_prefix("0x") {
                    Some(d) => (16, d),
                    None => (10, v.as_str()),
                };
                p.add_dma = u32::from_str_radix(digits, radix).unwrap_or(0);
            }
            let mut ops = Vec::new();
            match emit(&p, &mut ops) {
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
        }
        other => {
            eprintln!("unknown mode {other}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
