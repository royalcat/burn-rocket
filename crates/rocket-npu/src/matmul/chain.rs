//! The contiguous self-chaining regcmd layout for batched NPU submit.
//!
//! A multi-task NPU job can run as ONE hardware kick over CONTIGUOUS regcmds when each
//! task's trailer redirects the PC to the next task (an embedded `PC_BASE_ADDRESS` write
//! plus the next segment's `PC_REGISTER_AMOUNTS` length) and the kernel sets
//! `PC_TASK_CON.TASK_NUMBER = N` to gate a single completion IRQ. The `PC_BASE_ADDRESS`
//! redirect is LOAD-BEARING: with the address left pointing at task 0 the PC runs task 0
//! and stops, whatever the amount says (established by on-HW A/B on the RK1: contiguous
//! with the trailer left at 0, and encoded-amount-only, both stall — task 0 runs, the
//! counter sticks, the fence times out).
//!
//! Port of `src/rocket_chain.c`, kept as pure slice work so the layout can be gated on a
//! host without a device: the driver only ever sees the packed words.
//!
//! The chain rewrites exactly two words of a program: the trailer's `PC_REGISTER_AMOUNTS`
//! op and the inert `OP_NONE` filler immediately before it, which every emitter writes as
//! the trailer `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]`. The op count — and so
//! the chained stride — is unchanged. If that trailer shape ever drifts, the link is not
//! written and the batch runs task 0 into a multi-second fence timeout, so a miss is
//! reported as one loud diagnostic (once) rather than a mystery hang.
//!
//! Chaining is an EXTENSION here, not the native mode: the stock driver runs a job's
//! tasks one per IRQ, so [`enabled`] defaults to false and only `ROCKET_BATCH_SUBMIT=1`
//! turns it on — and only on a kernel that honors `DRM_ROCKET_JOB_BATCHED`.

// M1a lands the layout ahead of its caller (`matmul::Context` batching and the attention
// path in M1b/M5), so nothing in the library calls these yet; the unit tests at the
// bottom are the host gate. Drop this attribute with the first caller.
#![allow(dead_code)]

use std::sync::{Once, OnceLock};

use crate::device::uapi::DrmRocketTask;
use crate::error::{Error, Result};
use crate::log::{log_error, log_warn};
use crate::regcmd::desc::{NpuOp, OP_NONE, OP_REG_PC, PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS};

/// Whether contiguous chaining is requested (C `rkt_chain_enabled`).
///
/// `ROCKET_BATCH_SUBMIT` forces it either way; unset means the provider's default, which
/// is **off** here — `Device::batched_submit_native` is false for the mainline driver, so
/// the gapped per-task layout that every gate has always run stays the default.
///
/// Asking is not enough: chaining is a joint layout contract (userspace self-links the
/// regcmds, the kernel sets `TASK_NUMBER = task_count`). A kernel without the
/// batched-submit patch ignores `DRM_ROCKET_JOB_BATCHED` and runs the chained layout down
/// the per-task path, which corrupts or stalls the job — so an unsupported kernel
/// disables chaining with a warning instead of silently proceeding.
///
/// Resolved once per process, like the C code.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let want = crate::knobs::boolean("ROCKET_BATCH_SUBMIT").unwrap_or(false);
        if want && !crate::device::batched_submit_supported_global() {
            log_warn!(
                "ROCKET_BATCH_SUBMIT=1 but this kernel does not honor \
                 DRM_ROCKET_JOB_BATCHED (needs the rocket batched-submit patch; the driver \
                 must report >= 1.1). Running the stock per-task path instead."
            );
            return false;
        }
        want
    })
}

/// The chained per-task stride in u64 words: `regcmd_count` rounded up to a 128-bit
/// (2-word) boundary, which is the PC's per-task advance (C `rkt_chain_words`).
///
/// For the matmul emitters' even op counts this equals the count exactly, so a chain has
/// no padding. A caller mixing regcmd lengths in one job must keep chaining off and let
/// each task keep its own gapped slot.
pub fn words(regcmd_count: u32) -> usize {
    (regcmd_count as usize + 1) & !1
}

/// `PC_DATA_AMOUNT` encoding for a regcmd of `count` u64 ops — mirrors the kernel's
/// `PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((count + 1) / 2 - 1)`.
///
/// `count == 0` is not a valid program; it encodes 0 here rather than wrapping.
pub fn amount_encode(count: u32) -> u32 {
    count.div_ceil(2).saturating_sub(1)
}

/// The single loud diagnostic for a drifted trailer, logged once per process so a batch
/// of N tiles does not emit N copies.
fn trailer_fail(where_: &str) {
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        log_error!(
            "rocket_chain: {where_} found no [OP_NONE, PC_REGISTER_AMOUNTS] trailer to \
             rewrite — the regcmd trailer shape drifted, so the chained batch link was \
             NOT written and this submit will stall. Disable ROCKET_BATCH_SUBMIT (and \
             file a bug); run the chain layout gate."
        );
    });
}

/// Index of the trailer's `PC_REGISTER_AMOUNTS` op and its encoded amount. Scans from the
/// end so it finds the trailer and not an earlier PC write.
fn find_trailer_amount(rc: &[u64]) -> Option<(usize, u32)> {
    rc.iter()
        .enumerate()
        .rev()
        .map(|(j, w)| (j, NpuOp(*w)))
        .find(|(_, op)| op.target() == OP_REG_PC && op.reg() == PC_REGISTER_AMOUNTS)
        .map(|(j, op)| (j, op.value()))
}

/// Rewrite the trailer's amount field to `amount`; `None` if there is no such op.
fn set_trailer_amount(rc: &mut [u64], amount: u32) -> Option<usize> {
    let (j, _) = find_trailer_amount(rc)?;
    rc[j] = NpuOp::new(OP_REG_PC, PC_REGISTER_AMOUNTS, amount).0;
    Some(j)
}

/// Repurpose the inert `OP_NONE` filler just before the amount op into a
/// `PC_BASE_ADDRESS` write pointing at `next_addr`.
///
/// Refuses unless the slot really is the inert filler, rather than clobbering a real op.
fn set_trailer_base(rc: &mut [u64], next_addr: u32) -> bool {
    let Some((amt, _)) = find_trailer_amount(rc) else {
        return false;
    };
    if amt < 1 || NpuOp(rc[amt - 1]).target() != OP_NONE {
        return false;
    }
    rc[amt - 1] = NpuOp::new(OP_REG_PC, PC_BASE_ADDRESS, next_addr).0;
    true
}

/// Restore the `OP_NONE` filler, dropping the forward link. Idempotent: a slot that still
/// holds `OP_NONE` is left as-is.
fn clear_trailer_base(rc: &mut [u64]) -> bool {
    let Some((amt, _)) = find_trailer_amount(rc) else {
        return false;
    };
    if amt < 1 {
        return false;
    }
    rc[amt - 1] = NpuOp::new(OP_NONE, 0, 0).0;
    true
}

/// Does `[word_off, word_off + count)` fit `dst`, with every address encodable in the
/// regcmd's 32-bit fields?
///
/// Two separate questions, as in C `rkt_chain_slot_ok`: does the slot fit inside the
/// buffer at all, and is the buffer addressable from a 32-bit register field. The second
/// checks the whole slice rather than the slot: if its last byte is encodable then so is
/// every slot inside it, and the 32-bit rule stays written down in one place.
fn slot_ok(where_: &str, dst: &[u64], base_addr: u32, word_off: usize, count: u32) -> bool {
    let end = word_off.saturating_add(count as usize);
    if end > dst.len() {
        log_error!(
            "rocket_chain: {where_} wants words [{word_off},{end}) of a {}-word regcmd \
             buffer",
            dst.len()
        );
        return false;
    }
    if !dst.is_empty() {
        let last = base_addr as u64 + dst.len() as u64 * 8 - 1;
        if last >> 32 != 0 {
            log_error!(
                "rocket_chain: {where_}: the regcmd buffer's last byte ({last:#x}) leaves \
                 the low 4 GiB the regcmd's 32-bit address fields can encode"
            );
            return false;
        }
    }
    true
}

/// The IOVA of `word_off` inside a buffer at `base_addr`, or `None` when it would not fit
/// a 32-bit regcmd address field.
fn slot_addr(base_addr: u32, word_off: usize) -> Option<u32> {
    let addr = base_addr as u64 + word_off as u64 * 8;
    (addr >> 32 == 0).then_some(addr as u32)
}

/// Pack tile `nb`'s freshly generated regcmd (`src`, `count` u64 ops) into `dst` and fill
/// its descriptor `tasks[nb]` (C `rkt_chain_pack`).
///
/// When `chained`, the program goes at the contiguous [`words`] stride and its trailer is
/// linked to the next task (next address + next segment length); the last task of `tasks`
/// has no successor, so its link is cleared instead — the C code needs a separate
/// `rkt_chain_seal` for that only because it cannot see the batch length. Otherwise the
/// program goes at `gapped_stride` words and is copied byte-for-byte.
///
/// A chained batch must be uniform-length (the emitters' op count is data-independent per
/// dtype); a caller mixing regcmd lengths in one job must pass `chained = false`.
///
/// Returns an error when the slot does not fit, its address does not fit the 32-bit field,
/// or the trailer is not the shape the chain claims — in which case the caller must NOT
/// submit the batch.
#[allow(clippy::too_many_arguments)] // mirrors rkt_chain_pack's C signature
pub fn pack(
    chained: bool,
    dst: &mut [u64],
    base_addr: u32,
    tasks: &mut [DrmRocketTask],
    nb: usize,
    src: &[u64],
    count: u32,
    gapped_stride: usize,
) -> Result<()> {
    if nb >= tasks.len() {
        return Err(Error::shape("chain_pack"));
    }
    let stride = if chained { words(count) } else { gapped_stride };
    let word_off = nb.saturating_mul(stride);
    if !slot_ok("pack", dst, base_addr, word_off, count) {
        return Err(Error::shape("chain_pack"));
    }
    let addr = slot_addr(base_addr, word_off).ok_or_else(|| {
        log_error!("rocket_chain: pack slot {word_off} is not encodable from {base_addr:#x}");
        Error::shape("chain_pack")
    })?;

    if !chained {
        let count = count as usize;
        if src.len() < count {
            return Err(Error::shape("chain_pack"));
        }
        dst[word_off..word_off + count].copy_from_slice(&src[..count]);
        tasks[nb] = DrmRocketTask {
            regcmd: addr,
            regcmd_count: count as u32,
        };
        return Ok(());
    }

    let next_word_off = if nb + 1 == tasks.len() {
        word_off
    } else {
        word_off + stride
    };
    pack_at(dst, base_addr, word_off, next_word_off, src, count, count)?;
    tasks[nb] = DrmRocketTask {
        regcmd: addr,
        regcmd_count: count,
    };
    Ok(())
}

/// The same link at an explicit offset, for a chain whose programs are NOT all the same
/// length (C `rkt_chain_pack_at`).
///
/// `word_off` is where this program goes in `dst` and `next_word_off` where the PC should
/// continue; `next_word_off == word_off` marks the LAST program, whose forward link is
/// cleared instead (the hardware stops there on `TASK_NUMBER`, but a live address would
/// have the PC prefetch past the chain) and whose `next_count` is ignored.
///
/// BOTH trailer fields describe the NEXT segment, and a uniform-length chain cannot tell.
/// The kernel programs `PC_BASE_ADDRESS` and `PC_REGISTER_AMOUNTS` from task 0 alone, so
/// for every program after it the PREVIOUS program's trailer supplies the address to
/// fetch from AND how much to fetch — which is why `next_count` is a parameter. Get it
/// wrong and the chain fetches a short segment as a long one: the affected program
/// executes partially or not at all, with nothing to fault on. [`verify`] is the check.
///
/// Unlike the C `rkt_chain_pack_at`, this takes no descriptor array: fill the
/// [`DrmRocketTask`] yourself from `base_addr + word_off * 8` and `count`, as [`pack`]
/// does.
pub fn pack_at(
    dst: &mut [u64],
    base_addr: u32,
    word_off: usize,
    next_word_off: usize,
    src: &[u64],
    count: u32,
    next_count: u32,
) -> Result<()> {
    let last = next_word_off == word_off;
    let words = count as usize;
    if src.len() < words || !slot_ok("pack_at", dst, base_addr, word_off, count) {
        return Err(Error::shape("chain_pack_at"));
    }
    if !last && next_word_off.saturating_add(1) > dst.len() {
        log_error!(
            "rocket_chain: pack_at links to word {next_word_off}, past the {}-word regcmd \
             buffer",
            dst.len()
        );
        return Err(Error::shape("chain_pack_at"));
    }

    let slot = &mut dst[word_off..word_off + words];
    slot.copy_from_slice(&src[..words]);

    // The NEXT segment's length, because that is what this write is for. The last
    // program has no next, so its own count goes in and is never read.
    let amount = amount_encode(if last { count } else { next_count });
    if set_trailer_amount(slot, amount).is_none() {
        trailer_fail("pack_at (amount)");
        return Err(Error::shape("chain_pack_at"));
    }
    if last {
        if !clear_trailer_base(slot) {
            trailer_fail("pack_at (seal)");
            return Err(Error::shape("chain_pack_at"));
        }
    } else {
        let Some(next_addr) = slot_addr(base_addr, next_word_off) else {
            log_error!(
                "rocket_chain: pack_at link to word {next_word_off} is not encodable from \
                 {base_addr:#x}"
            );
            return Err(Error::shape("chain_pack_at"));
        };
        if !set_trailer_base(slot, next_addr) {
            trailer_fail("pack_at (base)");
            return Err(Error::shape("chain_pack_at"));
        }
    }
    Ok(())
}

/// Read back what was actually packed and check the chain is self-consistent
/// (C `rkt_chain_verify`).
///
/// For every task: that its trailer's `PC_BASE_ADDRESS` points at the NEXT task's regcmd
/// address and its `PC_REGISTER_AMOUNTS` encodes the NEXT task's real word count, and that
/// the last task's forward link is cleared.
///
/// This exists because [`pack_at`] takes `next_count` as a parameter and cannot check it.
/// Here the answer is available — the next program really is in `dst` — so the caller's
/// assertion becomes checkable. Call it after packing and before submit.
///
/// `stride` is the chained per-task stride in words ([`words`]). A non-zero stride also
/// asserts the contiguous layout itself: task `i` must start at word `i * stride` and fit
/// inside its slot. Pass 0 for a heterogeneous chain whose offsets are explicit.
///
/// CHAINED batches only: a gapped batch has no forward links to check, and every task
/// would read as one whose link was never written.
pub fn verify(dst: &[u64], base_addr: u32, tasks: &[DrmRocketTask], stride: usize) -> Result<()> {
    let n = tasks.len();
    if n == 0 {
        return Ok(());
    }

    for (i, task) in tasks.iter().enumerate() {
        let base = base_addr as u64;
        if (task.regcmd as u64) < base {
            log_error!(
                "rocket_chain: verify task {i} regcmd {:#x} is below the regcmd buffer base \
                 {base_addr:#x}",
                task.regcmd
            );
            return Err(Error::shape("chain_verify"));
        }
        let off_bytes = task.regcmd as u64 - base;
        if !off_bytes.is_multiple_of(8) {
            log_error!(
                "rocket_chain: verify task {i} regcmd {:#x} is not 8-byte aligned inside the \
                 buffer",
                task.regcmd
            );
            return Err(Error::shape("chain_verify"));
        }
        let word_off = (off_bytes / 8) as usize;
        let count = task.regcmd_count as usize;
        if word_off.saturating_add(count) > dst.len() {
            log_error!("rocket_chain: verify task {i} runs past the regcmd buffer");
            return Err(Error::shape("chain_verify"));
        }
        if stride != 0 {
            let slot_off = i.saturating_mul(stride);
            if word_off != slot_off {
                log_error!(
                    "rocket_chain: verify task {i} starts at word {word_off}, the contiguous \
                     stride {stride} puts it at {slot_off}"
                );
                return Err(Error::shape("chain_verify"));
            }
            if count > stride {
                log_error!(
                    "rocket_chain: verify task {i} is {count} words, past its {stride}-word slot"
                );
                return Err(Error::shape("chain_verify"));
            }
        }

        let slot = &dst[word_off..word_off + count];
        let Some((amt, amount)) = find_trailer_amount(slot) else {
            log_error!(
                "rocket_chain: verify task {i} has no [OP_NONE, PC_REGISTER_AMOUNTS] trailer"
            );
            return Err(Error::shape("chain_verify"));
        };
        if amt < 1 {
            log_error!("rocket_chain: verify task {i} has a trailer with no link slot");
            return Err(Error::shape("chain_verify"));
        }

        let link = NpuOp(slot[amt - 1]);
        let is_link = link.target() == OP_REG_PC && link.reg() == PC_BASE_ADDRESS;

        if i + 1 == n {
            // No successor: the forward link must have been cleared, or the PC prefetches
            // past the end of the chain.
            if is_link {
                log_error!(
                    "rocket_chain: verify last task {i} still carries a forward link to \
                     {:#x} — the chain was not sealed",
                    link.value()
                );
                return Err(Error::shape("chain_verify"));
            }
            continue;
        }

        if !is_link {
            log_error!(
                "rocket_chain: verify task {i} has no forward link; the chain would run \
                 task {i} and stall"
            );
            return Err(Error::shape("chain_verify"));
        }
        if link.value() != tasks[i + 1].regcmd {
            log_error!(
                "rocket_chain: verify task {i} links to {:#x} but task {} is at {:#x}",
                link.value(),
                i + 1,
                tasks[i + 1].regcmd
            );
            return Err(Error::shape("chain_verify"));
        }
        // The claim that cannot be checked at pack time: this task's trailer says how many
        // words to fetch for the NEXT program, and only here is the next program's real
        // length in hand.
        let want = amount_encode(tasks[i + 1].regcmd_count);
        if amount != want {
            log_error!(
                "rocket_chain: verify task {i} encodes next length {amount} ({} words) but \
                 task {} is {} words ({want})",
                (amount + 1) * 2,
                i + 1,
                tasks[i + 1].regcmd_count
            );
            return Err(Error::shape("chain_verify"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regcmd::desc::{OP_40, OP_ENABLE, PC};
    use crate::regcmd::fp16::{MatmulParams, emit};

    /// An arbitrary low IOVA for a synthetic regcmd buffer.
    const BASE: u32 = 0x1000_0000;
    /// The gapped per-task stride the matmul paths use.
    const GAPPED: usize = 128;

    /// A real fp16 tile program, so the trailer under test is the one the emitters write.
    fn program(m: u32, k: u32, n: u32) -> Vec<u64> {
        let mut ops = Vec::new();
        let count = emit(
            &MatmulParams::new(m, k, n, 0x1000, 0x2000, 0x3000),
            &mut ops,
        )
        .unwrap();
        assert_eq!(count, ops.len());
        ops.into_iter().map(|op| op.0).collect()
    }

    /// A synthetic program with the trailer shape the chain rewrite claims:
    /// `[.., OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]`. The body is filler.
    fn synth(count: usize) -> Vec<u64> {
        let mut ops = vec![NpuOp::new(PC, 0x4004, 0xE).0; count];
        ops[count - 4] = NpuOp::new(OP_NONE, 0, 0).0;
        ops[count - 3] = NpuOp::new(PC, PC_REGISTER_AMOUNTS, 0).0;
        ops[count - 2] = NpuOp::new(OP_40, 0, 0).0;
        ops[count - 1] = NpuOp::new(OP_ENABLE, 0, 0).0;
        ops
    }

    fn trailer_index(prog: &[u64]) -> usize {
        find_trailer_amount(prog).expect("the emitter's trailer").0
    }

    #[test]
    fn stride_and_amount_encoding() {
        // The chained stride is the count rounded up to 128 bits, and the emitter's op
        // count is even, so a chain has no padding.
        assert_eq!(words(126), 126);
        assert_eq!(words(127), 128);
        assert_eq!(words(0), 0);
        // PC_DATA_AMOUNT: (n + 1) / 2 - 1.
        assert_eq!(amount_encode(1), 0);
        assert_eq!(amount_encode(2), 0);
        assert_eq!(amount_encode(3), 1);
        assert_eq!(amount_encode(126), 62);
        assert_eq!(amount_encode(128), 63);
        let prog = program(64, 512, 64);
        assert_eq!(
            prog.len() % 2,
            0,
            "an odd op count would make the stride != advance"
        );
        assert_eq!(words(prog.len() as u32), prog.len());
    }

    #[test]
    fn the_emitter_writes_the_trailer_the_chain_claims() {
        let prog = program(4, 64, 16);
        let t = trailer_index(&prog);
        assert_eq!(t, prog.len() - 3, "trailer is the last PC write");
        assert_eq!(
            NpuOp(prog[t]).value(),
            0,
            "a single-task program's end marker is 0"
        );
        assert_eq!(NpuOp(prog[t - 1]).target(), OP_NONE, "the filler slot");
        assert_eq!(
            NpuOp(prog[t + 1]).target(),
            OP_40,
            "the magic after the amount"
        );
        assert_eq!(NpuOp(prog[t + 2]).target(), OP_ENABLE, "OP_ENABLE is last");
        // The op count is data-independent, so a batch shares one stride.
        for (m, k, n) in [(64, 512, 64), (4, 64, 16), (128, 256, 32)] {
            assert_eq!(program(m, k, n).len(), prog.len(), "{m}x{k}x{n}");
        }
    }

    #[test]
    fn contiguous_pack_links_each_task_to_the_next() {
        let prog = program(64, 512, 64);
        let count = prog.len();
        let stride = words(count as u32);
        let mut dst = vec![0xABu64; 3 * stride];
        let mut tasks = vec![DrmRocketTask::default(); 3];

        for nb in 0..3 {
            pack(
                true,
                &mut dst,
                BASE,
                &mut tasks,
                nb,
                &prog,
                count as u32,
                GAPPED,
            )
            .unwrap();
        }

        // Each task's first op landed at i*stride, and its descriptor names that IOVA.
        for (i, task) in tasks.iter().enumerate() {
            let first = NpuOp(dst[i * stride]);
            assert_eq!(
                (first.target(), first.reg()),
                (crate::regcmd::desc::DPU, 0x4004)
            );
            assert_eq!(task.regcmd, BASE + (i * stride * 8) as u32);
            assert_eq!(task.regcmd_count, count as u32);
        }

        // Every task but the last links to the next; the last is sealed.
        for i in 0..2 {
            let slot = &dst[i * stride..(i + 1) * stride];
            let t = trailer_index(slot);
            let link = NpuOp(slot[t - 1]);
            assert_eq!((link.target(), link.reg()), (OP_REG_PC, PC_BASE_ADDRESS));
            assert_eq!(link.value(), tasks[i + 1].regcmd);
            assert_eq!(NpuOp(slot[t]).value(), amount_encode(count as u32));
        }
        let last = &dst[2 * stride..];
        let t = trailer_index(last);
        assert_eq!(
            NpuOp(last[t - 1]).target(),
            OP_NONE,
            "the last link is cleared"
        );

        verify(&dst, BASE, &tasks, stride).expect("a correctly packed chain verifies");
    }

    #[test]
    fn a_non_chained_pack_leaves_the_program_byte_identical() {
        let prog = program(64, 512, 64);
        let count = prog.len();
        let mut dst = vec![0xABu64; 4 * GAPPED];
        let mut tasks = vec![DrmRocketTask::default(); 2];

        pack(
            false,
            &mut dst,
            BASE,
            &mut tasks,
            1,
            &prog,
            count as u32,
            GAPPED,
        )
        .unwrap();

        // The gapped slot holds the program verbatim: no trailer rewrite, no link.
        assert_eq!(&dst[GAPPED..GAPPED + count], &prog[..]);
        assert_eq!(tasks[1].regcmd, BASE + (GAPPED * 8) as u32);
        assert_eq!(tasks[1].regcmd_count, count as u32);
        // And the gap before it is untouched, so a short copy would be visible.
        assert!(dst[..GAPPED].iter().all(|&w| w == 0xAB));
    }

    #[test]
    fn pack_at_rewrites_only_the_two_trailer_words() {
        let prog = program(64, 512, 64);
        let count = prog.len();
        let mut dst = vec![0u64; 2 * count + 8];
        let mut tasks = vec![DrmRocketTask::default(); 2];

        pack_at(&mut dst, BASE, 0, count, &prog, count as u32, count as u32).unwrap();
        pack_at(
            &mut dst,
            BASE,
            count,
            count,
            &prog,
            count as u32,
            count as u32,
        )
        .unwrap();

        let t = trailer_index(&prog);
        for (slot_off, is_last) in [(0usize, false), (count, true)] {
            let slot = &dst[slot_off..slot_off + count];
            for (j, (&got, &want)) in slot.iter().zip(prog.iter()).enumerate() {
                if j == t - 1 || j == t {
                    continue; // the link slot and the amount slot are the chain's business
                }
                assert_eq!(got, want, "word {j} of task at {slot_off} changed");
            }
            assert_eq!(NpuOp(slot[t]).value(), amount_encode(count as u32));
            if is_last {
                assert_eq!(NpuOp(slot[t - 1]).target(), OP_NONE);
            } else {
                let link = NpuOp(slot[t - 1]);
                assert_eq!((link.target(), link.reg()), (OP_REG_PC, PC_BASE_ADDRESS));
                assert_eq!(link.value(), BASE + (count * 8) as u32);
            }
        }
        tasks[0] = DrmRocketTask {
            regcmd: BASE,
            regcmd_count: count as u32,
        };
        tasks[1] = DrmRocketTask {
            regcmd: BASE + (count * 8) as u32,
            regcmd_count: count as u32,
        };
        verify(&dst, BASE, &tasks, count).expect("the packed chain verifies");
    }

    #[test]
    fn heterogeneous_chain_verifies() {
        // Three programs of DIFFERENT lengths: the case pack_at exists for, and the only
        // one where `next_count` can disagree with reality.
        const COUNTS: [usize; 3] = [12, 20, 16];
        const OFFS: [usize; 3] = [0, 32, 64];
        let mut dst = vec![0u64; 96];
        let mut tasks = vec![DrmRocketTask::default(); 3];

        for i in 0..3 {
            let last = i == 2;
            let src = synth(COUNTS[i]);
            let next_off = if last { OFFS[i] } else { OFFS[i + 1] };
            let next_count = COUNTS[if last { i } else { i + 1 }] as u32;
            pack_at(
                &mut dst,
                BASE,
                OFFS[i],
                next_off,
                &src,
                COUNTS[i] as u32,
                next_count,
            )
            .unwrap();
            tasks[i] = DrmRocketTask {
                regcmd: BASE + (OFFS[i] * 8) as u32,
                regcmd_count: COUNTS[i] as u32,
            };
        }
        verify(&dst, BASE, &tasks, 0).expect("a heterogeneous chain verifies");

        // The packed trailers decoded WITHOUT the pack code: the kernel's encoding of the
        // next program's length, and the next program's IOVA.
        for (i, &count) in COUNTS.iter().enumerate() {
            let slot = &dst[OFFS[i]..OFFS[i] + count];
            let t = trailer_index(slot);
            let last = i == 2;
            let want_amount = ((if last { count } else { COUNTS[i + 1] }) as u32).div_ceil(2) - 1;
            assert_eq!(NpuOp(slot[t]).value(), want_amount, "task {i} amount");
            if last {
                assert_eq!(NpuOp(slot[t - 1]).target(), OP_NONE, "task {i} link slot");
            } else {
                let link = NpuOp(slot[t - 1]);
                assert_eq!((link.target(), link.reg()), (OP_REG_PC, PC_BASE_ADDRESS));
                assert_eq!(link.value(), BASE + (OFFS[i + 1] * 8) as u32);
            }
        }
    }

    #[test]
    fn a_wrong_next_count_is_caught() {
        const COUNTS: [usize; 3] = [12, 20, 16];
        const OFFS: [usize; 3] = [0, 32, 64];

        for bad_at in [0usize, 1] {
            let mut dst = vec![0u64; 96];
            let mut tasks = vec![DrmRocketTask::default(); 3];
            for i in 0..3 {
                let last = i == 2;
                let src = synth(COUNTS[i]);
                let next_off = if last { OFFS[i] } else { OFFS[i + 1] };
                // The lie: this task declares the next program longer than it is.
                let next_count = if i == bad_at {
                    COUNTS[i + 1] as u32 + 2
                } else if last {
                    COUNTS[i] as u32
                } else {
                    COUNTS[i + 1] as u32
                };
                pack_at(
                    &mut dst,
                    BASE,
                    OFFS[i],
                    next_off,
                    &src,
                    COUNTS[i] as u32,
                    next_count,
                )
                .unwrap();
                tasks[i] = DrmRocketTask {
                    regcmd: BASE + (OFFS[i] * 8) as u32,
                    regcmd_count: COUNTS[i] as u32,
                };
            }
            assert!(
                verify(&dst, BASE, &tasks, 0).is_err(),
                "verify accepted task {bad_at} declaring the wrong length for its successor"
            );
        }
    }

    #[test]
    fn an_unsealed_last_task_is_caught() {
        let src = synth(12);
        let mut dst = vec![0u64; 64];
        // A "last" program that still carries a live link to the head of the buffer.
        pack_at(&mut dst, BASE, 0, 0, &src, 12, 12).unwrap();
        let t = trailer_index(&src);
        dst[t - 1] = NpuOp::new(OP_REG_PC, PC_BASE_ADDRESS, BASE).0;
        let tasks = [DrmRocketTask {
            regcmd: BASE,
            regcmd_count: 12,
        }];
        assert!(verify(&dst, BASE, &tasks, 0).is_err());
    }

    #[test]
    fn a_task_outside_the_buffer_is_caught() {
        let src = synth(12);
        let mut dst = vec![0u64; 64];
        pack_at(&mut dst, BASE, 0, 0, &src, 12, 12).unwrap();
        let mut tasks = [DrmRocketTask {
            regcmd: BASE,
            regcmd_count: 12,
        }];
        verify(&dst, BASE, &tasks, 0).unwrap();

        // A slot whose words would run past the end of the buffer.
        tasks[0].regcmd = BASE + (dst.len() as u32 * 8) - 8;
        assert!(verify(&dst, BASE, &tasks, 0).is_err(), "past the end");
        // And one below the buffer base.
        tasks[0].regcmd = BASE - 8;
        assert!(verify(&dst, BASE, &tasks, 0).is_err(), "below the base");
        // A misaligned one is refused rather than read.
        tasks[0].regcmd = BASE + 4;
        assert!(verify(&dst, BASE, &tasks, 0).is_err(), "misaligned");
    }

    #[test]
    fn a_slot_past_the_buffer_is_refused_by_pack() {
        let prog = program(4, 64, 16);
        let count = prog.len();
        let mut dst = vec![0u64; count];
        let mut tasks = [DrmRocketTask::default(); 2];
        // Task 1 needs words [count, 2*count) of a count-word buffer.
        assert!(
            pack(
                true,
                &mut dst,
                BASE,
                &mut tasks,
                1,
                &prog,
                count as u32,
                GAPPED
            )
            .is_err(),
            "packing past the buffer must fail, not silently write nothing"
        );
        assert_eq!(
            tasks[1].regcmd_count, 0,
            "a failed pack fills no descriptor"
        );
    }

    #[test]
    fn a_stride_mismatch_is_caught() {
        let prog = program(64, 512, 64);
        let count = prog.len();
        let stride = words(count as u32);
        let mut dst = vec![0u64; 2 * stride];
        let mut tasks = vec![DrmRocketTask::default(); 2];
        for nb in 0..2 {
            pack(
                true,
                &mut dst,
                BASE,
                &mut tasks,
                nb,
                &prog,
                count as u32,
                GAPPED,
            )
            .unwrap();
        }
        verify(&dst, BASE, &tasks, stride).unwrap();
        // The same tasks checked against a stride that does not describe the layout.
        assert!(verify(&dst, BASE, &tasks, stride + 2).is_err());
        assert!(verify(&dst, BASE, &tasks, stride / 2).is_err());
    }

    #[test]
    fn a_buffer_above_the_32_bit_window_is_refused() {
        let prog = program(4, 64, 16);
        let count = prog.len();
        // A buffer whose last byte leaves the low 4 GiB: no 32-bit field can name it.
        let high = 0xFFFF_FFF0u32;
        let mut dst = vec![0u64; count];
        let mut tasks = [DrmRocketTask::default(); 1];
        assert!(
            pack(
                true,
                &mut dst,
                high,
                &mut tasks,
                0,
                &prog,
                count as u32,
                GAPPED
            )
            .is_err()
        );
    }
}
