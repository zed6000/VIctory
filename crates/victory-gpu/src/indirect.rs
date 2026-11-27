//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

/// The groups of a mesh dispatch record, shared by the single clamp and the
/// window's pass so the two cannot drift apart.
macro_rules! mesh_groups {
    () => {
        r#"
// Mesh dispatches: the console's are one-dimensional.
uint3 mesh_groups(uint3 g) {
    return uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
}
"#
    };
}

const SOURCE: &str = concat!(mesh_groups!(), r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint spread; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at, uint record) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 requested = args.Load3(at);
        uint3 g = min(requested, uint3(65535, 65535, 65535));
        uint row = 0;
        uint asked = requested.x;
        if (spread != 0 && asked > 65535 && requested.y != 0 && requested.z == 1) {
            // Drop records over the per-dispatch budget before rounding up.
            if (asked > GROUPS) {
                g = uint3(0, 0, 0);
            } else {
                uint rows = (asked + 65534) / 65535;
                row = (asked + rows - 1) / rows;
                g = uint3(row, g.y, rows);
            }
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        if (spread != 0) {
            dest.Store(record * 20, row);
            dest.Store(record * 20 + 4, asked);
            dest.Store3(record * 20 + 8, g);
        } else {
            dest.Store3(at, g);
        }
    } else {
        uint3 g = mesh_groups(args.Load3(at));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at, record);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            if (spread != 0 && record < records) {
                dest.Store3(record * 20 + 8, uint3(0, 0, 0));
            } else {
                dest.Store(at, 0);
            }
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#);

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    compile(SOURCE, s!("clamp"))
}

/// A clamp pipeline of `source` (named `name` in compiler messages).
fn compile(source: &str, name: windows::core::PCSTR) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    spread: bool,
) -> Option<ID3D12Resource> {
    // An open window first: its draws come before this, in order.
    end_window(list);
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out_stride = if spread { 20 } else { stride };
    let out = list.arguments(u64::from(out_stride) * u64::from(records.max(1)))?;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ClampSingle, 1);
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(spread)];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Whether consecutive indirect draws are clamped a window at a time
/// (`VICTORY_CLAMP_WINDOW=1`; off by default).
///
/// Without it, each indirect execution has its own clamp: a wait for all
/// earlier work, a compute dispatch, another wait, then the draw, so the
/// card goes from drawing to compute and back for every record and no
/// draw overlaps the next.
///
/// With it, an indirect draw opens a window: one clamp pass is recorded
/// where the single clamp would be, and the draws after it that may join
/// add their records to the pass's table (upload memory the CPU fills
/// while recording; the GPU reads it when the list runs). They read their
/// clamped records from the window's part of the list's argument buffer,
/// with no pass and no barrier of their own. The pass clamps every record
/// of the window in the order the draws come, charging the list's budget
/// in that order, exactly what the single clamps did one after another.
///
/// Moving a record's clamp before the window's earlier draws changes
/// nothing as long as none of them writes buffers. So a window ends (the
/// next indirect draw opens a new one, after a wait for all before) at:
/// any draw, dispatch or unordered-access clear other than a joining
/// indirect draw ([`end_window`] in `crate::draw`), an indirect dispatch,
/// an indirect draw whose shaders may store to memory (unordered views,
/// heap views, title memory), work a host library records, the list's
/// end; and a draw does not join when the table, the window's argument
/// bytes or its records are full, or its arguments lie in a host copy the
/// pass has no view of. Title copies between buffers run on the CPU at
/// submission, before the list, and stream output is not translated: no
/// other command of a list writes buffer copies.
pub(crate) fn windows_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CLAMP_WINDOW"))
}

/// Draws a window's table holds at most.
const WINDOW_ENTRIES: u32 = 64;
/// Bytes of clamped records a window holds at most (the list's argument
/// buffer is at least this long).
const WINDOW_BYTES: u64 = 1 << 16;
/// Records a window clamps at most.
const WINDOW_RECORDS: u32 = 4096;
/// Host copies a window's pass can read records from.
const WINDOW_SOURCES: usize = 4;
/// A table: a header of 4 words, then 8 words per draw.
const TABLE_BYTES: u64 = 16 + 32 * WINDOW_ENTRIES as u64;

const WINDOW_SOURCE: &str = concat!(mesh_groups!(), r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u1), UAV(u0)"
// The host copies records are read from, only read.
ByteAddressBuffer a0 : register(t0);
ByteAddressBuffer a1 : register(t1);
ByteAddressBuffer a2 : register(t2);
ByteAddressBuffer a3 : register(t3);
// Records, draws; then per draw (at 16 + 32 * draw) its first record in the
// window, records, kind, stride, source, the records' offset in it, their
// offset in `dest`. Draws in order; none without records.
ByteAddressBuffer table : register(t4);
RWByteAddressBuffer dest : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
uint word(uint source, uint at) {
    [branch] if (source == 0) {
        return a0.Load(at);
    }
    [branch] if (source == 1) {
        return a1.Load(at);
    }
    [branch] if (source == 2) {
        return a2.Load(at);
    }
    return a3.Load(at);
}
// The single clamp's bounds, word for word: the record of `kind` at `from`
// in `source` clamped into `dest` at `to`; its cost.
uint clamped(uint kind, uint source, uint from, uint to) {
    uint cost;
    if (kind <= 1) {
        uint count = min(word(source, from), 1u << 24);
        uint instances = min(word(source, from + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(to, uint2(count, instances));
        dest.Store2(to + 8, uint2(word(source, from + 8), word(source, from + 12)));
        if (kind == 1) {
            dest.Store(to + 16, word(source, from + 16));
        }
    } else if (kind == 2) {
        uint3 g = min(uint3(word(source, from), word(source, from + 4), word(source, from + 8)), uint3(65535, 65535, 65535));
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(to, g);
    } else {
        uint3 g = mesh_groups(uint3(word(source, from), word(source, from + 4), word(source, from + 8)));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(to, g);
    }
    return cost;
}
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// As the single clamp: one group, tiles of 64 records, the first lane
// charges each tile's records in order.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint records = table.Load(0);
    uint draws = min(table.Load(4), 64u);
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint cost = 0;
        uint to = 0;
        if (record < records) {
            uint draw = 0;
            for (uint k = 1; k < draws; k++) {
                if (table.Load(16 + 32 * k) <= record) {
                    draw = k;
                }
            }
            uint4 head = table.Load4(16 + 32 * draw);
            uint3 place = table.Load3(32 + 32 * draw);
            uint index = record - head.x;
            to = place.z + index * head.w;
            cost = clamped(head.z, place.x, place.y + index * head.w, to);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            dest.Store(to, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#);

fn window_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match compile(WINDOW_SOURCE, s!("clamp window")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no windowed argument clamp: each indirect draw clamped on its own");
                None
            }
        })
        .as_ref()
}

/// Whether a window was ever opened: until then (always, with the switch
/// off) ending and forgetting windows takes no lock.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The window a list has open, and the host copies its windows read.
#[derive(Default)]
pub(crate) struct Window {
    open: Option<Open>,
    /// Host GPU addresses of copies earlier windows of the list read
    /// (newest first): the next window's pass gets views of them too. The
    /// list's refreshes hold those copies until it is reset.
    recent: Vec<u64>,
}

struct Open {
    /// The list's argument buffer, indirect arguments until the window ends.
    out: ID3D12Resource,
    /// The table's CPU address (upload memory of the list).
    table: usize,
    /// The copies the pass reads: host GPU addresses, t0 to t3.
    sources: Vec<u64>,
    draws: u32,
    records: u32,
    bytes: u64,
}

/// Ends `list`'s window, if one is open: its argument buffer back to
/// common. Before any command that may write buffers.
pub(crate) fn end_window(list: &HostList) {
    if !OPENED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let open = list.clamp_window().open.take();
    if let Some(open) = open {
        done(list, &open.out);
    }
}

/// Forgets `list`'s windows (it was reset).
pub(crate) fn forget_windows(list: &HostList) {
    if OPENED.load(std::sync::atomic::Ordering::Relaxed) {
        *list.clamp_window() = Window::default();
    }
}

/// The clamped records of an indirect draw of `records` records of `kind`
/// `stride` bytes apart, `offset` bytes into the host copy at host GPU
/// address `source`, in `list`'s open window, or in one opened for it: the
/// argument buffer, the records' offset in it, and whether a window was
/// opened (its pass replaced the list's compute root signature and
/// pipeline). `None` if the draw cannot be clamped in a window; the caller
/// clamps it on its own ([`bound`]).
pub(crate) fn windowed(
    list: &HostList,
    source: u64,
    offset: u64,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64, bool)> {
    let bytes = u64::from(stride) * u64::from(records.max(1));
    if !source.is_multiple_of(4)
        || !offset.is_multiple_of(4)
        || !stride.is_multiple_of(4)
        || records > WINDOW_RECORDS
        || bytes > WINDOW_BYTES
        || offset.checked_add(bytes)? > u64::from(u32::MAX)
    {
        return None;
    }
    let mut window = list.clamp_window();
    let mut opened = false;
    if !window.open.as_ref().is_some_and(|o| o.fits(source, bytes, records)) {
        if let Some(old) = window.open.take() {
            done(list, &old.out);
        }
        let mut sources = vec![source];
        sources.extend(window.recent.iter().copied().filter(|&s| s != source).take(WINDOW_SOURCES - 1));
        window.recent.clone_from(&sources);
        OPENED.store(true, std::sync::atomic::Ordering::Relaxed);
        window.open = Some(open(list, sources)?);
        opened = true;
    }
    let o = window.open.as_mut()?;
    let at = o.bytes;
    if records > 0 {
        let index = o.sources.iter().position(|&s| s == source)? as u32;
        let entry = [o.records, records, kind, stride, index, offset as u32, at as u32, 0];
        // SAFETY: the table's words lie in the list's upload memory, alive
        // until it is reset; the GPU reads them only once the list runs.
        unsafe {
            let table = o.table as *mut u32;
            for (k, &w) in entry.iter().enumerate() {
                table.add(4 + 8 * o.draws as usize + k).write_volatile(w);
            }
            o.draws += 1;
            o.records += records;
            table.write_volatile(o.records);
            table.add(1).write_volatile(o.draws);
        }
    }
    o.bytes += bytes;
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::ClampWindowDraws, 1);
        if opened {
            crate::copystat::add(crate::copystat::C::ClampWindows, 1);
        }
    }
    Some((o.out.clone(), at, opened))
}

impl Open {
    /// Whether a draw of `records` records (`bytes` bytes) in the copy at
    /// `source` joins the window.
    fn fits(&self, source: u64, bytes: u64, records: u32) -> bool {
        self.draws < WINDOW_ENTRIES
            && self.bytes + bytes <= WINDOW_BYTES
            && self.records + records <= WINDOW_RECORDS
            && self.sources.contains(&source)
    }
}

/// Whether an indirect draw (as [`windowed`] takes it) joins `list`'s open
/// window (tests: to read a window's records before the next one).
#[cfg(test)]
fn joins(list: &HostList, source: u64, (_, stride, records): (u32, u32, u32)) -> bool {
    let bytes = u64::from(stride) * u64::from(records.max(1));
    list.clamp_window().open.as_ref().is_some_and(|o| o.fits(source, bytes, records))
}

/// Records a window's pass in `list`, reading records from `sources`.
fn open(list: &HostList, sources: Vec<u64>) -> Option<Open> {
    let (Some(c), Some(budget)) = (window_clamp(), list.budget()) else {
        return None;
    };
    let out = list.arguments(WINDOW_BYTES)?;
    let (table_gpu, table) = list.scratch(TABLE_BYTES)?;
    // SAFETY: the header of the list's fresh upload memory.
    unsafe { std::ptr::write_bytes(table, 0, 16) };
    let l = &list.list;
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // host copies the list's refreshes hold.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for k in 0..WINDOW_SOURCES {
            l.SetComputeRootShaderResourceView(k as u32, *sources.get(k).unwrap_or(&sources[0]));
        }
        l.SetComputeRootShaderResourceView(4, table_gpu);
        l.SetComputeRootUnorderedAccessView(5, budget);
        l.SetComputeRootUnorderedAccessView(6, out.GetGPUVirtualAddress());
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(Open { out, table: table as usize, sources, draws: 0, records: 0, bytes: 0 })
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, false).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), false).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// What a test list records, in order.
    #[derive(Clone, Copy)]
    enum Step {
        /// An indirect draw: title buffer, byte offset, kind, records.
        Draw(usize, u64, u32, u32),
        /// A shader of the title writes `value` at a byte offset of a title
        /// buffer (a draw or dispatch that may store: it ends the window).
        Write(usize, u64, u32),
        /// As `Write`, with the window left open: what a window that did not
        /// end there would clamp.
        WriteUnended(usize, u64, u32),
    }

    fn stride(kind: u32) -> u32 {
        match kind {
            0 => 16,
            1 => 20,
            _ => 12,
        }
    }

    const WRITE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0)"
cbuffer Word : register(b0) { uint at; uint value; };
RWByteAddressBuffer title : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { title.Store(at, value); }
"#;

    /// The open window's draws (offset in its buffer, bytes, offset in the
    /// results) and its buffer.
    type Pending = (Vec<(u64, u64, u64)>, Option<super::ID3D12Resource>);

    /// Records `steps` into one list over title buffers holding `init`,
    /// each draw clamped on its own or (`window`) in windows, and runs it:
    /// the words of each draw's clamped records. `None` without a host GPU.
    fn run(init: &[Vec<u32>], steps: &[Step], window: bool) -> Option<Vec<Vec<u32>>> {
        use super::{D3D12_RESOURCE_STATE_COMMON as COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE as COPY_SOURCE};
        use super::{D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT as ARGUMENTS, transition};
        let host = super::host()?;
        let titles: Vec<_> = init
            .iter()
            .map(|words| {
                let (buffer, data) = host.shared_buffer(words.len() as u64 * 4).unwrap();
                // SAFETY: our mapped buffer, written before the list runs; a
                // live buffer.
                let gpu = unsafe {
                    std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), words.len());
                    buffer.GetGPUVirtualAddress()
                };
                (buffer, gpu)
            })
            .collect();
        let (results, data) = host.shared_buffer(1 << 20).unwrap();
        let writer = super::compile(WRITE, super::s!("write")).unwrap();
        let list = host.new_list().unwrap();
        // Each draw's records: offset in `results`, words.
        let mut placed = Vec::new();
        let mut at_results = 0u64;
        let mut pending: Pending = (Vec::new(), None);
        // Ends the window and copies its records into `results`.
        let flush = |(draws, out): &mut Pending| {
            super::end_window(&list);
            if let (Some(out), false) = (out.as_ref(), draws.is_empty()) {
                // SAFETY: an open list, live buffers.
                unsafe {
                    list.list.ResourceBarrier(&[transition(out, COMMON, COPY_SOURCE)]);
                    for &(at, len, to) in draws.iter() {
                        list.list.CopyBufferRegion(&results, to, out, at, len);
                    }
                    list.list.ResourceBarrier(&[transition(out, COPY_SOURCE, COMMON)]);
                }
            }
            draws.clear();
        };
        for &step in steps {
            match step {
                Step::Draw(buffer, offset, kind, records) => {
                    let spec = (kind, stride(kind), records);
                    let len = u64::from(stride(kind)) * u64::from(records.max(1));
                    let source = titles[buffer].1;
                    if window {
                        if !super::joins(&list, source, spec) {
                            flush(&mut pending);
                        }
                        let (out, at, _) = super::windowed(&list, source, offset, spec).expect("a window");
                        pending.0.push((at, len, at_results));
                        pending.1 = Some(out);
                    } else {
                        let out = super::bound(&list, source + offset, spec, false).unwrap();
                        // SAFETY: an open list, live buffers.
                        unsafe {
                            list.list.ResourceBarrier(&[transition(&out, ARGUMENTS, COPY_SOURCE)]);
                            list.list.CopyBufferRegion(&results, at_results, &out, 0, len);
                            list.list.ResourceBarrier(&[transition(&out, COPY_SOURCE, ARGUMENTS)]);
                        }
                        super::done(&list, &out);
                    }
                    // An execution of no records reads none: nothing to
                    // compare (the single clamp leaves what was there).
                    placed.push((at_results, if records == 0 { 0 } else { len as usize / 4 }));
                    at_results += len.next_multiple_of(256);
                }
                Step::Write(buffer, at, value) | Step::WriteUnended(buffer, at, value) => {
                    if window && matches!(step, Step::Write(..)) {
                        flush(&mut pending);
                    }
                    let constants = [at as u32, value];
                    // SAFETY: an open list; our pipeline and a title buffer.
                    unsafe {
                        list.list.SetComputeRootSignature(&writer.root);
                        list.list.SetPipelineState(&writer.pso);
                        list.list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.list.SetComputeRootUnorderedAccessView(1, titles[buffer].1);
                        list.list.Dispatch(1, 1, 1);
                    }
                }
            }
        }
        if window {
            flush(&mut pending);
        }
        list.close();
        host.execute(&[&list]);
        let words = |&(at, n): &(u64, usize)| {
            // SAFETY: inside our mapped buffer; the list has run.
            unsafe { std::slice::from_raw_parts(data.add(at as usize).cast::<u32>(), n).to_vec() }
        };
        Some(placed.iter().map(words).collect())
    }

    /// The windows' records, after checking they are the single clamps'
    /// word for word.
    fn same(init: &[Vec<u32>], steps: &[Step]) -> Option<Vec<Vec<u32>>> {
        let single = run(init, steps, false)?;
        let windowed = run(init, steps, true)?;
        assert_eq!(single.len(), windowed.len());
        for (k, (a, b)) in single.iter().zip(&windowed).enumerate() {
            assert_eq!(a, b, "draw {k}: single {a:?}, windowed {b:?}");
        }
        Some(windowed)
    }

    /// Mesh dispatches, draws and indexed draws from two buffers, every
    /// seventh mesh record garbage (all ones) so that the budget runs out
    /// in the middle; more draws than a table holds, one of 100 records
    /// across two tiles and one of none: the windows clamp what the single
    /// clamps do.
    #[test]
    fn windows_clamp_as_single_clamps_do() {
        let mut a = Vec::new();
        for i in 0..400u32 {
            a.extend(if i % 7 == 3 { [u32::MAX; 3] } else { [100 + i, 1 + i % 3, i % 2] });
        }
        let draws = a.len() as u64 * 4;
        for i in 0..40u32 {
            a.extend(if i % 9 == 4 { [u32::MAX; 4] } else { [3 * i + 1, 2, i, 0] });
        }
        let indexed = a.len() as u64 * 4;
        for i in 0..40u32 {
            a.extend([6 * i, 1 + i, 7, 1, i]);
        }
        let b: Vec<u32> = (0..64u32).flat_map(|i| [70_000 + i, 2, 3]).collect();
        let mut steps = Vec::new();
        for i in 0..200u64 {
            steps.push(Step::Draw(0, 12 * i, 10, 1));
            if i % 10 == 5 {
                steps.push(Step::Draw(1, 12 * (i % 64), 10, 1));
                steps.push(Step::Draw(0, draws + 16 * (i % 40), 0, 1));
                steps.push(Step::Draw(0, indexed + 20 * (i % 40), 1, 1));
            }
            if i == 120 {
                steps.push(Step::Draw(0, 12 * 200, 10, 100));
                steps.push(Step::Draw(1, 0, 10, 0));
            }
        }
        let Some(out) = same(&[a, b], &steps) else {
            return; // no host GPU here
        };
        let ran = out.iter().filter(|r| r.first().is_some_and(|&w| w != 0)).count();
        assert!(ran > 50 && ran + 50 < out.len(), "the budget ran out in the middle: {ran} of {} ran", out.len());
        assert_eq!(out[0], [100, 1, 0], "a record as it is");
    }

    /// Mesh grids with more than one group along y or z (the case reported
    /// on #3197 with #2996: `(3, 2, 4)` read back as 24 groups clamped on
    /// its own, 3 in a window): a window keeps the single clamp's mesh
    /// rule, whatever that rule is, word for word.
    #[test]
    fn windowed_mesh_grids_are_the_single_clamps() {
        let a: Vec<u32> = [[3, 2, 4], [2, 3, 5], [65_535, 1, 1], [257, 256, 1], [3, 0, 4], [1, 1, 1]].concat();
        let steps: Vec<Step> = (0..6).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out[5], [1, 1, 1]);
    }

    /// A budget spent before the window opens stays spent: its draws all
    /// run empty, as they do clamped one by one.
    #[test]
    fn a_window_after_a_spent_budget_runs_empty() {
        let mut a: Vec<u32> = [u32::MAX; 3].repeat(20);
        a.extend([5, 1, 1].repeat(20));
        let mut steps: Vec<Step> = (0..20).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        // A store in between: the rest in a window of its own.
        steps.push(Step::Write(0, 0, u32::MAX));
        steps.extend((20..40).map(|i| Step::Draw(0, 12 * i, 10, 1)));
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out.iter().take(20).filter(|r| r[0] != 0).count(), 16, "16 garbage records fit");
        assert!(out[20..].iter().all(|r| r[0] == 0), "the spent budget empties the window");
    }

    /// A shader writing a record between two draws ends the window: the
    /// draw after it reads the record written, as clamped on its own. A
    /// window left open would have clamped the record from before.
    #[test]
    fn a_store_ends_the_window() {
        let a = [5, 1, 1].repeat(4);
        let steps = |write: fn(usize, u64, u32) -> Step| {
            vec![Step::Draw(0, 0, 10, 1), Step::Draw(0, 12, 10, 1), write(0, 24, 9), Step::Draw(0, 24, 10, 1), Step::Draw(0, 36, 10, 1)]
        };
        let Some(out) = same(std::slice::from_ref(&a), &steps(Step::Write)) else {
            return; // no host GPU here
        };
        assert_eq!(out[2], [9, 1, 1], "the record as written");
        assert_eq!(out[3], [5, 1, 1]);
        let unended = run(&[a], &steps(Step::WriteUnended), true).unwrap();
        assert_eq!(unended[2], [5, 1, 1], "a window over the store clamps the record from before it");
    }

    /// An execution of `max` records with a count buffer asking for more
    /// (the GPU runs min(count, max)): exactly `max` records are clamped,
    /// across two tiles, the garbage after them in the title's buffer is
    /// left out, and the next draw's records follow.
    #[test]
    fn a_window_clamps_max_records_whatever_the_count() {
        let mut a: Vec<u32> = (0..70u32).flat_map(|i| [10 + i, 1, 0, 0]).collect();
        a.extend([u32::MAX; 4].repeat(30));
        a.extend([4, 4, 0, 0]);
        let next = (a.len() as u64 - 4) * 4;
        let steps = [Step::Draw(0, 0, 0, 70), Step::Draw(0, next, 0, 1)];
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out[0].len(), 70 * 4);
        assert!((0..70).all(|i| out[0][4 * i] == 10 + i as u32));
        assert_eq!(out[1], [4, 4, 0, 0]);
    }

    /// Windows keep the argument buffer's states right: indirect arguments
    /// while open, common when the list closes, whatever order lists run in.
    #[test]
    fn windows_leave_the_argument_buffer_common() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 6]>().write_volatile([3, 1, 1, 4, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        issued();
        let recorded: Vec<_> = lists
            .iter()
            .map(|l| {
                let (out, _, opened) = super::windowed(l, gpu, 0, (10, 12, 1)).unwrap();
                assert!(opened);
                let (again, at, opened) = super::windowed(l, gpu, 12, (10, 12, 1)).unwrap();
                assert!(!opened && at == 12 && again.as_raw() == out.as_raw(), "the second draw joins");
                l.close();
                (out, issued())
            })
            .collect();
        host.execute(&[&lists[1], &lists[0]]);
        let mut run = issued();
        run.extend(recorded[1].1.clone());
        run.extend(recorded[0].1.clone());
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        let mut states: std::collections::HashMap<usize, u32> =
            [key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
        replay(&run, &mut states).unwrap();
        assert!(states.values().all(|&s| s == 0), "all common again");
        for (out, _) in &recorded {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours, the lists have run.
            let clamped = unsafe {
                out.Map(0, None, Some(&mut p)).unwrap();
                p.cast::<[u32; 6]>().read_volatile()
            };
            assert_eq!(clamped, [3, 1, 1, 4, 1, 1]);
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    #[test]
    fn spread_records_split_large_x_and_keep_small_unspread() {
        let Some(host) = super::host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        let cases = [
            ([115_200, 1, 1], [57_600, 115_200, 57_600, 1, 2]),
            ([65_535, 1, 1], [0, 65_535, 65_535, 1, 1]),
            ([65_536, 1, 1], [32_768, 65_536, 32_768, 1, 2]),
            ([65_537, 1, 1], [32_769, 65_537, 32_769, 1, 2]),
            ([70_000, 2, 1], [35_000, 70_000, 35_000, 2, 2]),
            ([100, 2, 1], [0, 100, 100, 2, 1]),
            ([65_537, 0, 1], [0, 65_537, 65_535, 0, 1]),
            ([65_537, 1, 0], [0, 65_537, 65_535, 1, 0]),
            ([80_000, 1, 2], [0, 80_000, 65_535, 1, 2]),
            ([10, 70_000, 1], [0, 10, 10, 65_535, 1]),
            ([70_000, 131_072, 1], [35_000, 70_000, 0, 0, 0]),
            ([65_536, 131_072, 1], [32_768, 65_536, 0, 0, 0]),
            ([u32::MAX, 1, 1], [0, u32::MAX, 0, 0, 0]),
        ];
        let records = cases.len() as u32;
        let (title, data) = host.shared_buffer(u64::from(records) * 12).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: the mapping covers one triple for each synthetic record.
        unsafe {
            for (i, (input, _)) in cases.iter().enumerate() {
                data.cast::<[u32; 3]>().add(i).write(*input);
            }
        }
        // SAFETY: title is a live D3D12 buffer.
        let address = unsafe { title.GetGPUVirtualAddress() };
        let out = super::bound(&list, address, (2, 12, records), true).unwrap();
        super::done(&list, &out);
        list.close();
        // Read back the clamp output. The original records are never dispatched.
        host.execute(&[&list]);
        // SAFETY: the output mapping covers five words per completed record.
        let got = unsafe {
            let mut mapped = std::ptr::null_mut();
            out.Map(0, None, Some(&mut mapped)).unwrap();
            let got = std::slice::from_raw_parts(mapped.cast::<[u32; 5]>(), cases.len()).to_vec();
            out.Unmap(0, None);
            got
        };
        for ((input, _), output) in cases.iter().zip(&got) {
            println!("spread record {input:?} -> {output:?}");
        }
        for ((input, expected), output) in cases.iter().zip(got) {
            assert_eq!(output, *expected, "spread record for {input:?}");
        }
    }
}
