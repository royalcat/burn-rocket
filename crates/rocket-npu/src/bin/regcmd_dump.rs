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
//!   regcmd_dump conv         IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE [OH OW]
//!   regcmd_dump emit-conv    IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE FP32TOFP16 IN WT OUT [DSY DSX]
//!   regcmd_dump emit-conv-act IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE KIND IN WT OUT
//!   regcmd_dump emit-conv-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT [DW_GROUP]
//!   regcmd_dump emit-conv-dw-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT BIAS \
//!        I_SCALE_BITS W_SCALE_BITS O_SCALE_BITS IN_ZP W_ZP OUT_ZP INT8_OUT [DW_GROUP]
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
//! `conv` prints `ops::conv::conv2d_plan_code` (0 or the negated reason). `emit-conv`
//! prints the word count and the program of `regcmd::conv::gen_conv2d_fp16` /
//! `gen_conv2d_dw_fp16` (DEPTHWISE selects the latter), with the optional deconvolution
//! interior-dilation strides DSY/DSX (1 = off); `emit-conv-act` fuses a smooth
//! activation through `ops::activation::lut_epilogue_build`. Both print `-1` when the
//! generator refuses, like the C oracle.
//!
//! `emit-int8` with ACCUMULATE=1 is not ported: the C int8 emitter has an
//! EW-add path there, but the C library itself documents that encoding as
//! non-bit-exact and never used in production, so the Rust port leaves it out.
//! `emit-int4` and `emit-tf32` ignore FP32TOFP16/ACCUMULATE/ADD_DMA, as the C's
//! plain-conv int4/tf32 generators do.

use std::process::ExitCode;

use rocket_npu::matmul::plan::{self, BF16, INT4, INT8, INT8_GW, Plan, TF32};
use rocket_npu::ops::activation::{Activation, LutTable, lut_epilogue_build};
use rocket_npu::ops::conv::{Conv2dDesc, conv2d_oh, conv2d_ow, conv2d_plan_code};
use rocket_npu::regcmd::conv::{
    ConvAct, ConvInt8Params, ConvParams, gen_conv2d_dw_fp16, gen_conv2d_dw_int8, gen_conv2d_fp16,
    gen_conv2d_int8,
};
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

/// The conv family: `conv` (the plan code), `emit-conv` / `emit-conv-act` (the program).
///
/// The argument layout is `tools/c-oracle/oracle`'s: 13 scalars after the mode, then the
/// mode's own tail (OH/OW for the plan, `FP32TOFP16|KIND IN WT OUT [DSY DSX]` for the
/// emitters).
fn conv_mode(mode: &str, args: &[String]) -> ExitCode {
    if args.len() < 15 {
        eprintln!("conv needs IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE");
        return ExitCode::FAILURE;
    }
    let n = |i: usize| args[i].parse::<usize>().unwrap_or(0);
    let mut d = Conv2dDesc {
        ic: n(2),
        ih: n(3),
        iw: n(4),
        oc: n(5),
        kh: n(6),
        kw: n(7),
        ..Default::default()
    };
    d.stride_y = n(8);
    d.stride_x = n(9);
    d.pad_top = n(10);
    d.pad_left = n(11);
    d.dil_y = n(12);
    d.dil_x = n(13);
    d.depthwise = n(14) != 0;

    if mode == "conv" {
        if args.len() > 15 {
            d.oh = n(15);
        }
        if args.len() > 16 {
            d.ow = n(16);
        }
        println!("{}", conv2d_plan_code(&d));
        return ExitCode::SUCCESS;
    }
    if args.len() < 19 {
        eprintln!("emit-conv needs FP32TOFP16|KIND IN WT OUT [DSY DSX]");
        return ExitCode::FAILURE;
    }
    let mut p = ConvParams::new(
        d.ic as u16,
        d.ih as u16,
        d.iw as u16,
        d.oc as u16,
        conv2d_oh(&d) as u16,
        conv2d_ow(&d) as u16,
        d.kh as u16,
        d.kw as u16,
        dma(&args[16]),
        dma(&args[17]),
        dma(&args[18]),
    );
    p.stride_y = d.stride_y as u8;
    p.stride_x = d.stride_x as u8;
    p.dil_y = d.dil_y as u8;
    p.dil_x = d.dil_x as u8;
    p.pad_top = d.pad_top as u8;
    p.pad_left = d.pad_left as u8;
    p.deconv_sy = args.get(19).map(|v| v.parse().unwrap_or(1)).unwrap_or(1);
    p.deconv_sx = args.get(20).map(|v| v.parse().unwrap_or(1)).unwrap_or(1);

    let mut ops = Vec::new();
    let r = if mode == "emit-conv-act" {
        let code: i32 = args[15].parse().unwrap_or(-1);
        let Some(kind) = Activation::from_code(code) else {
            println!("-1");
            eprintln!("unknown activation kind {code}");
            return ExitCode::SUCCESS;
        };
        let mut lut = LutTable::zeroed();
        match lut_epilogue_build(kind, &mut lut) {
            Ok(ep) => {
                p.fp32_to_fp16 = true;
                p.act = Some(ConvAct {
                    lut: lut.words(),
                    ep: &ep,
                });
                if d.depthwise {
                    gen_conv2d_dw_fp16(&mut p, &mut ops)
                } else {
                    gen_conv2d_fp16(&mut p, &mut ops)
                }
            }
            Err(e) => {
                println!("-1");
                eprintln!("{e}");
                return ExitCode::SUCCESS;
            }
        }
    } else {
        p.fp32_to_fp16 = args[15] != "0";
        if d.depthwise {
            gen_conv2d_dw_fp16(&mut p, &mut ops)
        } else {
            gen_conv2d_fp16(&mut p, &mut ops)
        }
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

/// The int8 family: `emit-conv-int8` (the direct int32-raw program) and
/// `emit-conv-dw-int8` (the depthwise int8-out requant, or its int32-raw diagnostic
/// form with `INT8_OUT = 0`).
///
/// The argument layout is `tools/c-oracle/oracle`'s: 12 geometry scalars, the three
/// IOVAs, then the quant tail. The scales are raw `f32` bit patterns in hex.
fn conv_int8_mode(mode: &str, args: &[String]) -> ExitCode {
    if args.len() < 17 {
        eprintln!("emit-conv-int8 needs IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT");
        return ExitCode::FAILURE;
    }
    let n = |i: usize| args[i].parse::<usize>().unwrap_or(0);
    let mut d = Conv2dDesc {
        ic: n(2),
        ih: n(3),
        iw: n(4),
        oc: n(5),
        kh: n(6),
        kw: n(7),
        stride_y: n(8),
        stride_x: n(9),
        pad_top: n(10),
        pad_left: n(11),
        dil_y: n(12),
        dil_x: n(13),
        ..Default::default()
    };
    d.depthwise = mode == "emit-conv-dw-int8";
    // The depthwise invariant OC == IC; a 0 in that argument means "the entry's own C"
    // (the library passes `oc = C`).
    if d.depthwise && d.oc == 0 {
        d.oc = d.ic;
    }
    let mut p = ConvInt8Params::new(
        d.ic as u16,
        d.ih as u16,
        d.iw as u16,
        d.oc as u16,
        conv2d_oh(&d) as u16,
        conv2d_ow(&d) as u16,
        d.kh as u16,
        d.kw as u16,
        dma(&args[14]),
        dma(&args[15]),
        dma(&args[16]),
    );
    p.stride_y = d.stride_y as u8;
    p.stride_x = d.stride_x as u8;
    p.dil_y = d.dil_y as u8;
    p.dil_x = d.dil_x as u8;
    p.pad_top = d.pad_top as u8;
    p.pad_left = d.pad_left as u8;

    let mut ops = Vec::new();
    let r = if !d.depthwise {
        p.dw_group = args.get(17).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
        gen_conv2d_int8(&mut p, &mut ops)
    } else {
        if args.len() < 24 {
            eprintln!("emit-conv-dw-int8 needs BIAS I W O SCALE_BITS IN_ZP W_ZP OUT_ZP INT8_OUT");
            return ExitCode::FAILURE;
        }
        p.bias_dma = dma(&args[17]);
        let s = |i: usize| f32::from_bits(dma(&args[i]));
        p.in_scale = s(18);
        p.w_scale = s(19);
        p.out_scale = s(20);
        let zp = |i: usize| args[i].parse::<i32>().unwrap_or(0) + 0x80;
        p.input_zero_point = zp(21);
        p.output_zero_point = zp(22);
        p.weight_zero_point = zp(23);
        p.int8_out = args.get(24).map(|v| v != "0").unwrap_or(true);
        p.dw_group = args.get(25).map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
        gen_conv2d_dw_int8(&mut p, &mut ops)
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
        eprintln!("       regcmd_dump conv IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE [OH OW]");
        eprintln!(
            "       regcmd_dump emit-conv IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE FP32TOFP16 IN WT OUT [DSY DSX]"
        );
        eprintln!(
            "       regcmd_dump emit-conv-act IC IH IW OC KH KW SY SX PT PL DY DX DEPTHWISE KIND IN WT OUT"
        );
        eprintln!(
            "       regcmd_dump emit-conv-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT [DW_GROUP]"
        );
        eprintln!(
            "       regcmd_dump emit-conv-dw-int8 IC IH IW OC KH KW SY SX PT PL DY DX IN WT OUT BIAS I_SCALE_BITS W_SCALE_BITS O_SCALE_BITS IN_ZP W_ZP OUT_ZP INT8_OUT [DW_GROUP]"
        );
        return ExitCode::FAILURE;
    }
    let mode = args[1].as_str();

    if mode == "conv" || mode == "emit-conv" || mode == "emit-conv-act" {
        return conv_mode(mode, &args);
    }

    if mode == "emit-conv-int8" || mode == "emit-conv-dw-int8" {
        return conv_int8_mode(mode, &args);
    }

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
