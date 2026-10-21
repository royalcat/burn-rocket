//! Per-dtype matrix geometry.

/// How one dtype maps onto the NPU's cube layout and planner constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Bytes per element in the host-packed operands.
    pub elem_bytes: usize,
    /// K alignment: the planner rejects `K % k_align != 0`.
    pub k_align: u32,
    /// N alignment.
    pub n_align: u32,
    /// The weight K-group.
    pub k_group: u32,
    /// The weight N-group.
    pub n_group: u32,
    /// Whether the dtype runs through the int8 (W8A8) context.
    pub int8: bool,
}

pub const FP16: Geometry = Geometry {
    elem_bytes: 2,
    k_align: 8,
    n_align: 8,
    k_group: 32,
    n_group: 16,
    int8: false,
};
