//! Dump planner and regcmd output for the C oracle comparison.
//!
//! Usage:
//!   regcmd_dump plan         M K N
//!   regcmd_dump plan-int8    M K N
//!   regcmd_dump plan-int8-gw M K N [GROUP]
//!   regcmd_dump plan-int4    M K N
//!   regcmd_dump plan-tf32    M K N
//!   regcmd_dump plan-bf16    M K N
//!   regcmd_dump emit         M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-int8    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-int4    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-tf32    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump emit-bf16    M K N IN_DMA WT_DMA OUT_DMA [FP32TOFP16] [ACCUMULATE] [ADD_DMA]
//!   regcmd_dump recip        K
//!   regcmd_dump pool         C H W
//!   regcmd_dump emit-pool    C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN_DMA OUT_DMA
//!
//! The emitters take the TILE dims (Mt Kt Nt), not the full problem dims, and a
//! plan prints `Mt Kt Nt jobs` or `-1`, matching `tools/c-oracle/oracle`.
//!
//! `recip` prints the fp16 average-pool reciprocal bits of K
//! (`regcmd::pool::ppu_recip_kernel_fp16`). `pool` prints the spatial-reduce
//! decomposition for one (C, H, W): the `global_avgpool_plan_code` reason code, then
//! one `H <n> f...` / `W <n> f...` line with `reduce_factor_axis`'s factors (`-1` for an
//! axis that is not 16-smooth). `emit-pool` prints the word count and the program of
//! `regcmd::pool::gen_pool_fp16`, or `-1` when the generator refuses the params (the C
//! oracle prints the same portable rejection line).
//!
//! `emit-int8` with ACCUMULATE=1 is not ported: the C int8 emitter has an
//! EW-add path there, but the C library itself documents that encoding as
//! non-bit-exact and never used in production, so the Rust port leaves it out.
//! `emit-int4` and `emit-tf32` ignore FP32TOFP16/ACCUMULATE/ADD_DMA, as the C's
//! plain-conv int4/tf32 generators do.

use std::process::ExitCode;

use rocket_npu::matmul::plan::{self, BF16, INT4, INT8, INT8_GW, Plan, TF32};
use rocket_npu::regcmd::fp16::{MatmulParams, emit};
use rocket_npu::regcmd::pool::{PoolMethod, PoolParams, gen_pool_fp16};

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
        "emit-int4" => rocket_npu::regcmd::int4::emit(params, &mut ops),
        "emit-tf32" => rocket_npu::regcmd::tf32::emit(params, &mut ops),
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: regcmd_dump plan|plan-int8|plan-int8-gw|plan-int4|plan-tf32|plan-bf16 M K N [GROUP]"
        );
        eprintln!(
            "       regcmd_dump emit|emit-int8|emit-int4|emit-tf32|emit-bf16 M K N IN WT OUT [FP32TOFP16] [ACCUMULATE] [ADD_DMA]"
        );
        eprintln!("       regcmd_dump recip K");
        eprintln!("       regcmd_dump pool C H W");
        eprintln!(
            "       regcmd_dump emit-pool C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN OUT"
        );
        return ExitCode::FAILURE;
    }
    let mode = args[1].as_str();

    if mode == "recip" {
        let k: String = args[2].clone();
        let k = dma(&k);
        println!("{:08x}", rocket_npu::regcmd::pool::ppu_recip_kernel_fp16(k));
        return ExitCode::SUCCESS;
    }

    if mode == "pool" {
        if args.len() < 5 {
            eprintln!("pool needs C H W");
            return ExitCode::FAILURE;
        }
        let dim = |i: usize| args[i].parse::<usize>().unwrap_or(0);
        let (c, h, w) = (dim(2), dim(3), dim(4));
        println!(
            "{}",
            rocket_npu::ops::reduce::global_avgpool_plan_code(c, h, w)
        );
        let mut f = [0usize; rocket_npu::ops::reduce::REDUCE_MAX_PASS];
        for (axis, n) in [("H", h), ("W", w)] {
            match rocket_npu::ops::reduce::reduce_factor_axis(n, &mut f) {
                Ok(count) => {
                    print!("{axis} {count}");
                    for v in &f[..count] {
                        print!(" {v}");
                    }
                    println!();
                }
                Err(_) => println!("{axis} -1"),
            }
        }
        return ExitCode::SUCCESS;
    }

    if mode == "emit-pool" {
        if args.len() < 20 {
            eprintln!("emit-pool needs C IH IW OH OW KH KW SY SX PT PL PB PR METHOD RW RH IN OUT");
            return ExitCode::FAILURE;
        }
        let num = |i: usize| args[i].parse::<u32>().unwrap_or(0);
        let method = match num(15) {
            0 => PoolMethod::Avg,
            1 => PoolMethod::Max,
            2 => PoolMethod::Min,
            other => {
                eprintln!("bad pooling method {other}");
                return ExitCode::FAILURE;
            }
        };
        let p = PoolParams {
            c: num(2) as u16,
            ih: num(3) as u16,
            iw: num(4) as u16,
            oh: num(5) as u16,
            ow: num(6) as u16,
            kh: num(7) as u8,
            kw: num(8) as u8,
            stride_y: num(9) as u8,
            stride_x: num(10) as u8,
            pad_top: num(11) as u8,
            pad_left: num(12) as u8,
            pad_bottom: num(13) as u8,
            pad_right: num(14) as u8,
            method,
            recip_w: dma(&args[16]),
            recip_h: dma(&args[17]),
            input_dma: dma(&args[18]),
            output_dma: dma(&args[19]),
        };
        let mut ops = Vec::new();
        match gen_pool_fp16(&p, &mut ops) {
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
        return ExitCode::SUCCESS;
    }

    let Some((m, k, n)) = parse_dims(&args) else {
        eprintln!("bad dimensions");
        return ExitCode::FAILURE;
    };

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
        "plan-int4" => {
            print_plan(&INT4, 0, m, k, n);
            ExitCode::SUCCESS
        }
        "plan-tf32" => {
            print_plan(&TF32, 0, m, k, n);
            ExitCode::SUCCESS
        }
        "emit" | "emit-int8" | "emit-tf32" | "emit-bf16" => {
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
        "emit-int4" => {
            if args.len() < 8 {
                eprintln!("emit needs M K N IN WT OUT");
                return ExitCode::FAILURE;
            }
            // The int4 generator takes no fp32-out/accumulate/add args.
            let p = MatmulParams::new(
                m as u32,
                k as u32,
                n as u32,
                dma(&args[5]),
                dma(&args[6]),
                dma(&args[7]),
            );
            emit_program(mode, &p)
        }
        other => {
            eprintln!("unknown mode {other}");
            ExitCode::FAILURE
        }
    }
}
