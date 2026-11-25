//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). Grand Theft Auto
        // VI keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 32-bit float ones stand
/// in, so that depth its shaders write through unordered views
/// (`crate::twin`) arrives as the very float a draw rasterises; copied into
/// 24-bit depth it never tested EQUAL against a draw on the RX 6700 XT.
/// shortcut: a depth bias on such a target counts in float steps, not
/// 1/65535; scale it if a title's 16-bit depth biased draws show acne.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 20,
        119 => 21,
        120 => 22,
        other => other,
    }
}

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 64),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    pub(crate) format: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) row_pitch: u32,
}

impl PlacedFootprint {
    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }

    /// The byte ranges (from the footprint's start) that a `size` texel
    /// region at texel `at` covers: one per row of blocks; `None` if the
    /// format has no size or the region leaves the footprint.
    pub(crate) fn rows(&self, at: [u32; 3], size: [u32; 3]) -> Option<Vec<(u64, u64)>> {
        let (element, block) = element(self.format)?;
        let blocks = |texels: u32| u64::from(texels).div_ceil(block);
        let [x, y, z] = at.map(u64::from);
        let (w, h, d) = (blocks(size[0]), blocks(size[1]), u64::from(size[2]));
        let (x, y) = (x / block, y / block);
        if x + w > blocks(self.width) || y + h > blocks(self.height) || z + d > u64::from(self.depth) {
            return None;
        }
        let pitch = u64::from(self.row_pitch);
        let slice = pitch * blocks(self.height);
        Some(
            (z..z + d)
                .flat_map(|k| (y..y + h).map(move |r| (k * slice + r * pitch + x * element, w * element)))
                .collect(),
        )
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's upload footprint (offset 0), row count and unpadded row size.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let mip = index % mips;
    let plane = index / mips / slices;
    let depth_stencil = matches!(desc.format, 19..=22 | 44..=47);
    // Each plane contains all the mip levels and array slices. Depth and
    // stencil copies use separate formats and 512-byte row alignment.
    let (format, pitch_alignment) = match (depth_stencil, plane) {
        (true, 0) => (39, 512), // DXGI_FORMAT_R32_TYPELESS
        (true, 1) => (60, 512), // DXGI_FORMAT_R8_TYPELESS
        (false, 0) => (desc.format, PITCH),
        _ => return None,
    };
    let (element, block) = element(format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(pitch_alignment)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Places subresources at 512-byte boundaries in an upload buffer.
/// Row pitch is aligned to 512 bytes for depth/stencil planes and 256 bytes
/// otherwise. Unknown sizes give the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (Grand Theft Auto VI places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007), made on first use
    /// ([`Resource::host`]); buffers do not yet. `None`: not made (yet, or
    /// given up by [`evict`]); `Some(None)`: the host refused it.
    #[cfg(windows)]
    host: std::sync::Mutex<Option<Option<std::sync::Arc<HostTexture>>>>,
    /// The creation's initial state and clear value, for the host texture.
    #[cfg(windows)]
    made_with: (u32, Option<[u32; 5]>),
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
///
/// One per texture memory (decision 0007, "One host texture per texture
/// memory"): title textures at the same address with the same shape and
/// element size share it, each viewing it in its own format.
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    /// Formats views may take besides the texture's own family: the host
    /// took them as castable formats when it made the texture.
    pub(crate) castable: Vec<u32>,
    /// A host copy or resolve wrote it: its contents are the host's, not
    /// title memory's (`crate::texture`).
    pub(crate) host_written: std::sync::atomic::AtomicBool,
    /// Its uploads from title memory (`crate::texture`), made on first use.
    pub(crate) staging: std::sync::Mutex<Option<std::sync::Arc<crate::texture::Staging>>>,
    /// A depth texture's writable twin (`crate::twin`), made with its first
    /// unordered access view; `None` inside if it has none.
    pub(crate) twin: std::sync::OnceLock<Option<crate::twin::Twin>>,
    /// Bytes it takes on the host GPU.
    bytes: u64,
    /// A list used it (counted once).
    used: std::sync::atomic::AtomicBool,
    /// A shader-visible heap slot viewed it (diagnostics: [`texture_census`]).
    pub(crate) viewed: std::sync::atomic::AtomicBool,
    /// The memory it copies, for title textures fed from it.
    key: Option<MemoryKey>,
    /// Its range in a heap of textures (`crate::placement`), given back
    /// when it goes; `None` for a committed texture.
    placement: Option<crate::placement::Placement>,
    /// When the title last used it: presented frame and milliseconds since
    /// the first stamp ([`HostTexture::stamp`]).
    last: [std::sync::atomic::AtomicU64; 2],
}

#[cfg(windows)]
impl HostTexture {
    /// A host texture made outside the title's resources (twins).
    pub(crate) fn new(
        texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        state: u32,
        castable: Vec<u32>,
    ) -> Self {
        let bytes = crate::host::host().map_or(0, |h| {
            // SAFETY: plain queries on a live resource.
            unsafe { h.device.GetResourceAllocationInfo(0, &[texture.GetDesc()]) }.SizeInBytes
        });
        TEXTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TEXTURE_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        HostTexture {
            texture,
            state: state.into(),
            castable,
            host_written: false.into(),
            staging: std::sync::Mutex::new(None),
            twin: std::sync::OnceLock::new(),
            bytes,
            used: false.into(),
            viewed: false.into(),
            key: None,
            placement: None,
            last: [crate::queue::presented().into(), now_ms().into()],
        }
    }

    /// The title used it now: a list of the title's names it, or a shader
    /// read a heap slot viewing it (`crate::descriptor::make_read`).
    pub(crate) fn stamp(&self) {
        self.stamp_at(crate::queue::presented(), now_ms());
    }

    /// Used at presented frame `frame`, `ms` milliseconds in.
    fn stamp_at(&self, frame: u64, ms: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.last[0].store(frame, Relaxed);
        self.last[1].store(ms, Relaxed);
    }

    /// When the title last used it: presented frame and milliseconds
    /// (tests).
    #[cfg(test)]
    pub(crate) fn last_used(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.last[0].load(Relaxed), self.last[1].load(Relaxed))
    }

    /// Forgets its uses (tests: a stamp is then seen to happen).
    #[cfg(test)]
    pub(crate) fn unstamp(&self) {
        self.stamp_at(NEVER, 0);
    }

    /// Whether the title used it in this presented frame or the last one
    /// (made then, named by a list of its own, or read through the heap as
    /// the last pass found).
    pub(crate) fn used_recently(&self) -> bool {
        let frame = self.last[0].load(std::sync::atomic::Ordering::Relaxed);
        frame != NEVER && frame.saturating_add(1) >= crate::queue::presented()
    }

    /// A list uses it: counted the first time (diagnostics of how much of
    /// the host textures' memory the GPU ever touches).
    pub(crate) fn note_used(&self) {
        if !self.used.swap(true, std::sync::atomic::Ordering::Relaxed) {
            USED.fetch_add(self.bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Bytes of host textures some list used (while they live).
#[cfg(windows)]
static USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host textures alive and the bytes they take (the fps line's companion).
#[cfg(windows)]
static TEXTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(windows)]
static TEXTURE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Title textures that found their memory's host texture made already.
#[cfg(windows)]
static SHARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes the title textures that shared would have taken in host textures
/// of their own.
#[cfg(windows)]
static SAVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Title textures whose memory's host texture could not take them (more
/// uses, a format it cannot be viewed in): a new one replaced it.
#[cfg(windows)]
static REPLACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host textures alive, their bytes and the bytes of those lists used,
/// title textures that shared one (and the bytes that saved), those that
/// replaced their memory's.
#[cfg(windows)]
pub(crate) fn texture_counts() -> (u64, u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let shared = (SHARED.load(Relaxed), SAVED.load(Relaxed), REPLACED.load(Relaxed));
    (TEXTURES.load(Relaxed), TEXTURE_BYTES.load(Relaxed), USED.load(Relaxed), shared.0, shared.1, shared.2)
}

/// Bytes of the live host textures of title textures by what wrote them
/// (index 0 the GPU: render target, depth or unordered access; 1 title
/// memory) and how they were reached (0 by lists, 1 only by heap slots, 2
/// neither).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn texture_census() -> [[u64; 3]; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    let mut bytes = [[0; 3]; 2];
    let map = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    for t in map.iter().flat_map(|m| m.values()).filter_map(std::sync::Weak::upgrade) {
        let gpu = usize::from(!crate::texture::title_written(&t.texture));
        let reach = if t.used.load(Relaxed) { 0 } else if t.viewed.load(Relaxed) { 1 } else { 2 };
        bytes[1 - gpu][reach] += t.bytes;
    }
    bytes
}

#[cfg(windows)]
impl Drop for HostTexture {
    fn drop(&mut self) {
        if let Some(p) = self.placement.take() {
            crate::placement::free(self.texture.clone(), p);
        }
        TEXTURES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        TEXTURE_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
        if *self.used.get_mut() {
            USED.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        {
            use windows::core::Interface;
            life::note(self.texture.as_raw() as usize, life::LET_GO, 0, life::holders(&self.texture));
        }
    }
}

/// Diagnostics: the last events in host textures' lives, for
/// device-removed reports that find a fault inside a freed one (host
/// textures are named with their address, `@0x...`).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod life {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// The title released the resource the texture copies.
    pub(crate) const RELEASED: u8 = 1;
    /// A view heap slot (the value) stopped viewing it.
    pub(crate) const SLOT: u8 = 2;
    /// The shared host copy (title resource and lists) let go of it.
    pub(crate) const LET_GO: u8 = 3;
    /// A submission started (value: its number; no texture).
    pub(crate) const SUBMITTED: u8 = 4;
    /// That submission's lists have run.
    pub(crate) const RAN: u8 = 5;

    /// (texture, time, kind, value, references left after the event).
    type Event = (usize, std::time::Instant, u8, u64, u32);
    static EVENTS: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());
    const KEPT: usize = 1 << 18;

    /// The references `texture` has besides the one about to go.
    pub(crate) fn holders(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> u32 {
        use windows::core::Interface;
        // SAFETY: a live COM object; one reference taken and given back.
        unsafe {
            let raw = texture.as_raw();
            let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
            ((**unknown).AddRef)(raw);
            ((**unknown).Release)(raw).saturating_sub(1)
        }
    }

    /// `left`: the texture's references once the event's own is gone.
    pub(crate) fn note(texture: usize, kind: u8, value: u64, left: u32) {
        let mut events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
        if events.len() >= KEPT {
            events.pop_front();
        }
        events.push_back((texture, std::time::Instant::now(), kind, value, left));
    }

    /// What happened to the texture named `name` (`...@0x<address>`), with
    /// the submissions around it, times relative to now.
    pub(crate) fn of(name: &str) -> String {
        let Some(at) = name
            .rsplit_once("@0x")
            .and_then(|(_, a)| usize::from_str_radix(a.split(|c: char| !c.is_ascii_hexdigit()).next()?, 16).ok())
        else {
            return String::new();
        };
        let now = std::time::Instant::now();
        let events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = Vec::new();
        let mut last_submission = None;
        for &(t, when, kind, value, left) in events.iter() {
            let ago = now.duration_since(when).as_secs_f64() * 1e3;
            match kind {
                SUBMITTED | RAN => last_submission = Some((kind, value, ago)),
                _ if t == at => {
                    if let Some((k, n, a)) = last_submission.take() {
                        let what = if k == SUBMITTED { "started" } else { "ran" };
                        text.push(format!("(submission {n} {what} {a:.1} ms ago)"));
                    }
                    let what = match kind {
                        RELEASED => "the title released its resource".to_owned(),
                        SLOT => format!("view heap slot {value} stopped viewing it"),
                        _ => "the shared host copy let go".to_owned(),
                    };
                    let freed = match (kind, left) {
                        (SLOT, 0) => ", its last holder (freed after the submission running then)",
                        (LET_GO, 0) => ", FREED",
                        _ => "",
                    };
                    text.push(format!("{ago:.1} ms ago {what}{freed}"));
                }
                _ => {}
            }
        }
        if let Some((k, n, a)) = last_submission {
            let what = if k == SUBMITTED { "started" } else { "ran" };
            text.push(format!("(last: submission {n} {what} {a:.1} ms ago)"));
        }
        format!(" [{name}: {}]", text.join("; "))
    }
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = resource(this).host()?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one (made on first
/// use). Owned: [`evict`] may give the host texture up meanwhile.
#[cfg(windows)]
pub(crate) fn host_resource(this: This) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host().map(|h| h.texture.clone())
}

/// A title texture's host copy if it is made, without making it: for work
/// that is no use of the title's (uploads of what heap slots view).
#[cfg(windows)]
pub(crate) fn made_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host.lock().unwrap_or_else(|e| e.into_inner()).clone().flatten()
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host()
}

/// The host description of a title texture: the same, without the
/// console's layout and flags, in a format the host takes for its use.
#[cfg(windows)]
fn host_desc(desc: &ResourceDesc) -> Option<ResourceDesc> {
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; Grand Theft Auto VI's shadow maps and material depth ask
    // for it): unordered views of such a texture write its twin
    // (`crate::twin`).
    if d.flags & DEPTH != 0 {
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    Some(d)
}

const DEPTH: u32 = 2;
#[cfg(windows)]
const UNORDERED: u32 = 4;
/// Render target, depth, unordered access: what a host texture allows.
#[cfg(windows)]
const USES: u32 = 0x7;

/// What a texture memory is, for the one host texture it gets: its address
/// and shape (the console's layout resolved, mips counted, and the bytes
/// and texels of an element), and whether it is a depth texture (which
/// cannot share with colour ones on the host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MemoryKey {
    address: u64,
    dimension: u32,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mips: u32,
    samples: u32,
    swizzle: u32,
    element: (u64, u64),
    depth: bool,
}

/// The texture memory `desc` describes at `address`; `None` for buffers
/// and formats without a known element.
pub(crate) fn memory_key(address: u64, desc: &ResourceDesc) -> Option<MemoryKey> {
    if desc.dimension == BUFFER {
        return None;
    }
    Some(MemoryKey {
        address,
        dimension: desc.dimension,
        width: desc.width,
        height: desc.height,
        depth_or_array: desc.depth_or_array_size,
        mips: mip_count(desc),
        samples: desc.sample_count.max(1),
        swizzle: texture_layout(desc).map_or(desc.layout, |t| t.planes[0].layout.desc.swizzle.0),
        element: element(desc.format)?,
        depth: desc.flags & DEPTH != 0,
    })
}

/// The formats a host texture of `format` is made castable to: every
/// format of the same element (bytes and block texels), as the host's
/// format list casting allows (uncompressed among uncompressed, block
/// formats among block formats of the same block size), but depth ones
/// (refused in lists on the RX 6700 XT and WARP alike) and typeless ones
/// (no view takes them).
pub(crate) fn castable(format: u32) -> Vec<u32> {
    const DEPTH_FAMILIES: [u32; 10] = [19, 20, 21, 22, 40, 44, 45, 46, 47, 55];
    let Some(e) = element(format) else {
        return Vec::new();
    };
    (1..=115)
        .filter(|&f| f != format && element(f) == Some(e) && !DEPTH_FAMILIES.contains(&f))
        .filter(|&f| !(family(f) == f && family(f + 1) == f))
        .collect()
}

/// Live host textures by texture memory: title textures (resource objects
/// and placed views) of one memory share one.
#[cfg(windows)]
static MEMORY: std::sync::Mutex<
    Option<std::collections::HashMap<MemoryKey, std::sync::Weak<HostTexture>>>,
> = std::sync::Mutex::new(None);

/// The host texture of the memory at `address` that `desc` describes: the
/// live one if it takes `desc` (its uses, a format it can be viewed in),
/// else a new one, which becomes the memory's.
#[cfg(windows)]
fn shared_texture(
    address: u64,
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
) -> Option<std::sync::Arc<HostTexture>> {
    let d = host_desc(desc)?;
    let key = memory_key(address, desc);
    let mut map = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(Default::default);
    let live = key.and_then(|k| map.get(&k)?.upgrade());
    if let Some(t) = &live
        && takes(t, &d)
    {
        SHARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(host) = crate::host::host() {
            // SAFETY: a plain query on a host description (same layout).
            let info = unsafe { host.device.GetResourceAllocationInfo(0, &[std::ptr::from_ref(&d).cast::<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC>().read_unaligned()]) };
            SAVED.fetch_add(info.SizeInBytes, std::sync::atomic::Ordering::Relaxed);
        }
        return live;
    }
    // Host textures are made on first use, in any order: the new one takes
    // the uses of every title texture of the memory, so none replaces it.
    let mut d = d;
    if let Some(k) = key {
        let at = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
        for r in at.get(&address).into_iter().flatten().map(|&r| resource(r as This)) {
            if memory_key(address, &r.desc) == Some(k) {
                d.flags |= host_desc(&r.desc).map_or(0, |h| h.flags & USES);
            }
        }
    }
    // shortcut: title textures of a replaced host texture keep it (placed
    // views move to the new one, `texture_at`); copies (`alias_source`)
    // bridge the two where the shapes allow.
    let mut t = make_texture(&d, state, clear)?;
    t.key = key;
    let t = std::sync::Arc::new(t);
    if let Some(k) = key {
        remade(&k);
        if live.is_some() {
            REPLACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(?k, "a texture memory's host texture replaced by one with more uses or another format");
        }
        if map.len().is_power_of_two() {
            map.retain(|_, w| w.strong_count() > 0);
        }
        map.insert(k, std::sync::Arc::downgrade(&t));
    }
    Some(t)
}

/// Whether host texture `t` takes a title texture of host description
/// `d` at its memory: its uses, and a view format of its family or
/// castable list.
#[cfg(windows)]
fn takes(t: &HostTexture, d: &ResourceDesc) -> bool {
    // SAFETY: a live resource.
    let td = unsafe { t.texture.GetDesc() };
    let (flags, own) = (td.Flags.0 as u32, td.Format.0 as u32);
    d.flags & USES & !flags == 0
        && (family(d.format) == family(own)
            || t.castable.iter().any(|&c| family(c) == family(d.format)))
}

/// Whether `this`'s host texture is still its memory's (not replaced by one
/// with more uses).
#[cfg(windows)]
fn current(this: This) -> bool {
    let r = resource(this);
    let made = r.host.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let (Some(Some(h)), Some(k)) = (made, memory_key(r.data() as u64, &r.desc)) else {
        return true;
    };
    let map = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|m| m.get(&k)?.upgrade())
        .is_none_or(|live| std::sync::Arc::ptr_eq(&live, &h))
}

/// A host texture of host description `d` in GPU memory, castable to
/// [`castable`]'s formats (colour textures, through the enhanced-barrier
/// creation; it starts in the common state) where the host takes that.
#[cfg(windows)]
fn make_texture(d: &ResourceDesc, state: u32, clear: *const c_void) -> Option<HostTexture> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    use windows::core::Interface;
    let host = crate::host::host()?;
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = pc_format(value[0]);
        value
    });
    let clear_value = clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast::<D3D12_CLEAR_VALUE>());
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    let mut castable_formats = if d.flags & DEPTH == 0 { castable(d.format) } else { Vec::new() };
    let mut initial = crate::host::legal_for(d.flags, crate::host::texture_state(state));
    // In a heap of textures if it fits there (`crate::placement`).
    // SAFETY: the description has the host layout (checked size).
    let plain = unsafe { std::ptr::from_ref(d).cast::<D3D12_RESOURCE_DESC>().read_unaligned() };
    let mut placement = None;
    if clear_value.is_none()
        && crate::placement::fits(&host.device, &plain)
        && let Some((heap, p)) = crate::placement::place(&host.device, &plain)
    {
        let formats: Vec<DXGI_FORMAT> = castable_formats.iter().map(|&f| DXGI_FORMAT(f as i32)).collect();
        let made = match host.device.cast::<ID3D12Device10>() {
            Ok(device) if !formats.is_empty() => {
                let desc1 = desc1_of(&plain);
                // SAFETY: a range of the heap for this description; the
                // list of formats lives through the call.
                unsafe {
                    device.CreatePlacedResource2(&heap, p.offset(), &desc1, D3D12_BARRIER_LAYOUT_COMMON, None, Some(&formats), &mut texture)
                }
                .map(|()| 0)
            }
            _ => {
                // SAFETY: as above.
                unsafe {
                    host.device.CreatePlacedResource(&heap, p.offset(), &plain, D3D12_RESOURCE_STATES(initial as i32), None, &mut texture)
                }
                .map(|()| {
                    castable_formats.clear();
                    initial
                })
            }
        };
        match made {
            Ok(state) => {
                initial = state;
                placement = Some(p);
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(error = %e, desc = ?d, "a texture not placed in a heap: committed"));
                crate::placement::unplace(p);
                texture = None;
            }
        }
    }
    if texture.is_none()
        && !castable_formats.is_empty()
        && let Ok(device) = host.device.cast::<ID3D12Device10>()
    {
        let desc1 = desc1_of(&plain);
        let formats: Vec<DXGI_FORMAT> = castable_formats.iter().map(|&f| DXGI_FORMAT(f as i32)).collect();
        // SAFETY: a host description and our out pointer; the list and the
        // clear value live through the call.
        let made = unsafe {
            device.CreateCommittedResource3(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc1,
                D3D12_BARRIER_LAYOUT_COMMON,
                clear_value,
                None::<&ID3D12ProtectedResourceSession>,
                Some(&formats),
                &mut texture,
            )
        };
        if let Err(e) = made {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(error = %e, desc = ?d, "the host refuses castable formats: textures of one memory in other formats get their own host texture");
            });
            texture = None;
        } else {
            // The common layout is the common state.
            initial = 0;
        }
    }
    if texture.is_none() {
        castable_formats.clear();
        // SAFETY: the description has the host layout (checked size, same
        // fields); the out pointer is ours.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                std::ptr::from_ref(d).cast(),
                D3D12_RESOURCE_STATES(initial as i32),
                clear_value,
                &mut texture,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, desc = ?d, "no host texture");
        }
    }
    let texture = texture?;
    crate::host::name(&texture, || {
        // Its address, for the lifetime notes (`life`) of a freed one.
        format!("texture {}x{} f{} flags {:#x} @{:#x}", d.width, d.height, d.format, d.flags, texture.as_raw() as usize)
    });
    let initial = crate::host::legal_state(&texture, initial);
    let mut t = HostTexture::new(texture, initial, castable_formats);
    t.placement = placement;
    Some(t)
}

/// A plain description plus the sampler feedback region (none).
#[cfg(windows)]
fn desc1_of(p: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC1 {
    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC1 {
        Dimension: p.Dimension,
        Alignment: p.Alignment,
        Width: p.Width,
        Height: p.Height,
        DepthOrArraySize: p.DepthOrArraySize,
        MipLevels: p.MipLevels,
        Format: p.Format,
        SampleDesc: p.SampleDesc,
        Layout: p.Layout,
        Flags: p.Flags,
        SamplerFeedbackMipRegion: Default::default(),
    }
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A texture's description as words (the layout of placed views'
/// resource descriptions); `None` for buffers.
pub(crate) fn texture_desc(this: This) -> Option<[u32; 14]> {
    let (_, desc) = texture_memory(this)?;
    let mut words = [0u32; 14];
    // SAFETY: 56 bytes in the description's layout.
    unsafe { words.as_mut_ptr().cast::<ResourceDesc>().write_unaligned(desc) };
    Some(words)
}

/// A texture's format as the title gave it.
pub(crate) fn title_format(this: This) -> Option<u32> {
    (!this.is_null()).then(|| resource(this).desc.format)
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    /// Its host texture, made the first time it is asked for: a list uses
    /// the texture, a heap slot views it, a copy or upload reaches it. Most
    /// of the title's textures are never used (decision 0007, "Host
    /// textures made on first use").
    #[cfg(windows)]
    fn host(&self) -> Option<std::sync::Arc<HostTexture>> {
        if let Some(made) = self.host.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return made;
        }
        // Made without the lock held (`shared_texture` takes the memory
        // maps, which `evict` holds while it takes this lock).
        let (state, clear) = self.made_with;
        let clear = clear.as_ref().map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast());
        let made = shared_texture(self.data() as u64, &self.desc, state, clear);
        // A texture made meanwhile by another thread is the memory's too.
        self.host.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(made).clone()
    }

    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

/// A title texture's host texture if it is made.
#[cfg(windows)]
fn made(r: &Resource) -> Option<std::sync::Arc<HostTexture>> {
    r.host.lock().unwrap_or_else(|e| e.into_inner()).clone().flatten()
}

/// Milliseconds since the first call: host textures' use stamps.
#[cfg(windows)]
fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// A host texture fed from title memory that the title has not used for
/// this long (and this many presented frames) is given up ([`evict`]).
/// Two minutes: of the textures given up after 20 s, nine in ten were
/// read again within a minute (run evict4).
#[cfg(windows)]
pub(crate) const IDLE_MS: u64 = 120_000;
/// Frames as well as time: in slow runs (seconds a frame) 20 s alone cycled
/// the textures of a cutscene's alternating shots (run evict2), and 30
/// frames gave up what the menus and loading made (at 30 fps) before the
/// world read it (run evict3). At 30 fps the idle time decides; slower, the
/// card's budget does.
#[cfg(windows)]
pub(crate) const IDLE_FRAMES: u64 = 300;
/// A use stamp's frame for "never" (tests).
#[cfg(windows)]
const NEVER: u64 = u64::MAX;
/// Past this share of the card's memory budget, host textures fed from
/// title memory are given up least recently used first, until the card's
/// use is back under [`BUDGET_LOW`].
#[cfg(windows)]
const BUDGET_HIGH: f64 = 0.90;
#[cfg(windows)]
const BUDGET_LOW: f64 = 0.85;
/// A texture made again within this many frames of being given up counts
/// as remade soon ([`EVICTIONS`]): too eager an [`IDLE_MS`].
#[cfg(windows)]
const SOON_FRAMES: u64 = 10;

/// Host textures given up, their bytes, those made again since, and those
/// made again within [`SOON_FRAMES`].
#[cfg(windows)]
pub(crate) static EVICTIONS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Memories whose host texture was given up, with the frame it was.
#[cfg(windows)]
static GIVEN_UP: std::sync::Mutex<Option<std::collections::HashMap<MemoryKey, u64>>> = std::sync::Mutex::new(None);

/// A host texture is made for memory `k`: counted if it had been given up.
#[cfg(windows)]
fn remade(k: &MemoryKey) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut given = GIVEN_UP.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(at) = given.as_mut().and_then(|g| g.remove(k)) {
        EVICTIONS[2].fetch_add(1, Relaxed);
        if crate::queue::presented().saturating_sub(at) <= SOON_FRAMES {
            EVICTIONS[3].fetch_add(1, Relaxed);
        }
    }
}

/// Whether host texture `t` may be given up: its contents come from title
/// memory alone (no render target, depth or unordered access, no host copy
/// or resolve wrote it), so a texture made again has them back.
#[cfg(windows)]
fn evictable(t: &HostTexture) -> bool {
    t.key.is_some() && crate::texture::title_written(&t.texture) && !t.host_written.load(std::sync::atomic::Ordering::Relaxed)
}

/// Once per presented frame, before a submission (`Host::execute`, one at a
/// time, the last one has run): gives up host textures fed from title memory
/// that the title has not used for [`IDLE_MS`], and past [`BUDGET_HIGH`] of
/// the card's budget the least recently used ones until [`BUDGET_LOW`].
#[cfg(windows)]
pub(crate) fn evict_unused() {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
    let frame = crate::queue::presented();
    if LAST.swap(frame, Relaxed) == frame {
        return;
    }
    let over = crate::host::video_memory().map_or(0, |[(used, budget), _]| {
        // Free space in the heaps of textures takes the card's memory but
        // is not in use: new textures go there first.
        let used = used - crate::placement::free_bytes() as f64 / 1e6;
        if used > BUDGET_HIGH * budget { ((used - BUDGET_LOW * budget) * 1e6) as u64 } else { 0 }
    });
    evict(frame, now_ms(), over);
}

/// A host texture and the title textures (resource objects) holding it.
#[cfg(windows)]
pub(crate) type Owned = (std::sync::Arc<HostTexture>, Vec<usize>);

/// Gives up the host textures fed from title memory that the title has
/// not used for [`IDLE_MS`] and [`IDLE_FRAMES`] at (`frame`, `now` ms),
/// then least recently used ones until `over` bytes are gone
/// ([`give_up`]). Returns how many and their bytes.
#[cfg(windows)]
fn evict(frame: u64, now: u64, over: u64) -> (u64, u64) {
    // Read through the heap since the last pass: used (a slot's read mark is
    // reported as new only once, so its first read alone stamped it; run
    // evict1 gave up and made again hundreds of textures a frame).
    let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
    evict_among(&candidates(), &read, frame, now, over)
}

/// [`evict`] among `candidates`, with `read` (raw pointers) the host
/// textures shaders read through the heap since the last pass.
#[cfg(windows)]
pub(crate) fn evict_among(
    candidates: &[Owned],
    read: &std::collections::HashSet<usize>,
    frame: u64,
    now: u64,
    over: u64,
) -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    for (t, _) in candidates {
        if read.contains(&(windows::core::Interface::as_raw(&t.texture) as usize)) {
            t.stamp_at(frame, now);
        }
    }
    let uses: Vec<(u64, u64, u64)> =
        candidates.iter().map(|(t, _)| (t.last[0].load(Relaxed), t.last[1].load(Relaxed), t.bytes)).collect();
    let chosen = choose(&uses, frame, now, over);
    let gone: Vec<Owned> = chosen.into_iter().map(|i| candidates[i].clone()).collect();
    give_up(&gone, frame)
}

/// The host textures that may be given up ([`evictable`]), each with the
/// title textures holding it.
#[cfg(windows)]
fn candidates() -> Vec<Owned> {
    let at = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
    let mut by: std::collections::HashMap<usize, Owned> = std::collections::HashMap::new();
    for &r in at.values().flatten() {
        if let Some(t) = made(resource(r as This)).filter(|t| evictable(t)) {
            by.entry(std::sync::Arc::as_ptr(&t) as usize).or_insert_with(|| (t, Vec::new())).1.push(r);
        }
    }
    by.into_values().collect()
}

/// Of textures last used at (frame, ms) taking (bytes) each, those to give
/// up at (`frame`, `now`): every one idle for [`IDLE_FRAMES`] and
/// [`IDLE_MS`], then the least recently used others until `over` bytes are
/// gone (indices).
fn choose(uses: &[(u64, u64, u64)], frame: u64, now: u64, over: u64) -> Vec<usize> {
    let idle = |&(f, ms, _): &(u64, u64, u64)| frame.saturating_sub(f) >= IDLE_FRAMES && now.saturating_sub(ms) >= IDLE_MS;
    let (mut gone, mut rest): (Vec<usize>, Vec<usize>) = (0..uses.len()).partition(|&i| idle(&uses[i]));
    let mut over = over.saturating_sub(gone.iter().map(|&i| uses[i].2).sum());
    rest.sort_by_key(|&i| uses[i].1);
    for i in rest {
        if over == 0 {
            break;
        }
        over = over.saturating_sub(uses[i].2);
        gone.push(i);
    }
    gone
}

/// Gives up host textures `gone` at `frame`: first every heap slot viewing
/// one gets a null view (made again when a draw or a shader reads it), then
/// the title textures let go of it. Lists recorded before hold it until
/// they are reset, so no GPU work reads it freed; the next use makes it
/// again from title memory.
/// shortcut: a use racing this (a title thread recording) may view the
/// texture made again in a list that moved the one given up; the pass
/// gives up only textures unused for a while, or the least recently used.
#[cfg(windows)]
fn give_up(gone: &[Owned], frame: u64) -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if gone.is_empty() {
        return (0, 0);
    }
    {
        use windows::core::Interface;
        let raws = gone.iter().map(|(t, _)| t.texture.as_raw() as usize).collect();
        crate::descriptor::evict(&raws);
    }
    let at = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
    let live: std::collections::HashSet<usize> = at.values().flatten().copied().collect();
    let mut given = GIVEN_UP.lock().unwrap_or_else(|e| e.into_inner());
    let given = given.get_or_insert_with(Default::default);
    given.retain(|_, &mut f| frame.saturating_sub(f) <= SOON_FRAMES);
    let bytes = gone.iter().map(|(t, _)| t.bytes).sum();
    for (t, owners) in gone {
        for &r in owners.iter().filter(|r| live.contains(r)) {
            let mut h = resource(r as This).host.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(&*h, Some(Some(x)) if std::sync::Arc::ptr_eq(x, t)) {
                *h = None;
            }
        }
        if let Some(k) = t.key {
            given.insert(k, frame);
        }
    }
    EVICTIONS[0].fetch_add(gone.len() as u64, Relaxed);
    EVICTIONS[1].fetch_add(bytes, Relaxed);
    (gone.len() as u64, bytes)
}

/// Gives up `this`'s host texture as [`evict`] would (tests); false if it
/// has none or may not be given up.
#[cfg(all(windows, test))]
pub(crate) fn give_up_now(this: This) -> bool {
    owned(this).is_some_and(|o| give_up(&[o], crate::queue::presented()).0 == 1)
}

/// `this`'s host texture with the title textures holding it, if it may be
/// given up (tests: [`evict_among`] over textures of their own).
#[cfg(all(windows, test))]
pub(crate) fn owned(this: This) -> Option<Owned> {
    let t = made(resource(this))?;
    candidates().into_iter().find(|(c, _)| std::sync::Arc::ptr_eq(c, &t))
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear);
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: std::sync::Mutex::new(None),
        // The clear value names the format too: format, then color or
        // depth and stencil (5 words), kept for when the texture is made.
        #[cfg(windows)]
        // SAFETY: the title passes a whole clear value (or null).
        made_with: (state, (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() })),
    });
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    } else if host_desc(&desc).is_some() {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        let mut map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(resource.data() as u64).or_default().push(at);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

/// Title textures with a host copy, by the console memory they start at.
/// Title textures of one memory share a host texture where the host can
/// view it in each one's format; where it cannot (block-compressed against
/// plain formats, a cast the host refused, a host texture replaced),
/// [`alias_source`] finds the one the GPU writes.
#[cfg(windows)]
static TEXTURES_AT: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<usize>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// For a texture made for a placed view (`texture_at`): a title texture
/// the GPU writes (render target, depth or unordered access) at the same
/// memory, in the same layout and with the same bytes per row and rows
/// (elements of the same size, block-compressed ones counted in blocks),
/// that has another host texture: what the GPU wrote there lives in that
/// one, to be copied over (`texture::alias_copy`).
/// shortcut: same start only; overlapping ranges, other mips or slices of
/// one memory, and views of buffers are not matched.
#[cfg(windows)]
pub(crate) fn alias_source(placed: This) -> Option<This> {
    const WRITTEN: u32 = 0x7; // render target, depth, unordered access
    let b = resource(placed);
    let map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = |d: &ResourceDesc| {
        let (size, block) = element(d.format)?;
        let swizzle = texture_layout(d).map_or(d.layout, |t| t.planes[0].layout.desc.swizzle.0);
        let rows = u64::from(d.height).div_ceil(block);
        Some((d.dimension, d.width.div_ceil(block) * size, rows, d.depth_or_array_size, d.sample_count, swizzle))
    };
    let want = bytes(&b.desc)?;
    map.get(&(b.data() as u64))?
        .iter()
        .map(|&r| r as This)
        .filter(|&r| r != placed)
        .find(|&r| {
            let a = resource(r);
            a.desc.flags & WRITTEN != 0
                && bytes(&a.desc) == Some(want)
                && matches!((made(a), made(b)), (Some(x), Some(y)) if !std::sync::Arc::ptr_eq(&x, &y))
        })
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if made(self).is_some_and(|h| std::sync::Arc::strong_count(&h) > 2)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some(h) = made(self) {
            use windows::core::Interface;
            life::note(h.texture.as_raw() as usize, life::RELEASED, 0, life::holders(&h.texture));
        }
        crate::bindless::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension != BUFFER {
            let mut map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
            let at = self.data() as u64;
            if let Some(list) = map.get_mut(&at) {
                list.retain(|&r| r != std::ptr::from_ref::<Resource>(self) as usize);
                if list.is_empty() {
                    map.remove(&at);
                }
            }
        }
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from Grand Theft Auto VI's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From Grand Theft Auto VI's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, Grand Theft Auto VI's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// shortcut: never released (one host texture per distinct address and
/// description viewed); release them when the title unmaps or reuses the
/// memory once streaming grows VRAM use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Console bytes of the placed textures [`texture_at`] made (they are never
/// released).
#[cfg(windows)]
pub(crate) static PLACED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Console bytes of a texture of resource description `desc` (as words).
#[cfg(windows)]
pub(crate) fn placed_size(desc: &[u32; 14]) -> Option<u64> {
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    (d.dimension != BUFFER).then(|| footprint(&d).map(|f| f.size)).flatten()
}

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    // Words 1 and 13 are padding (stack bytes in the title's records).
    let mut desc = *desc;
    (desc[1], desc[13]) = (0, 0);
    let desc = &desc;
    // A resource whose memory's host texture was replaced since (one with
    // more uses) is made again, sharing the new one.
    // shortcut: the old one is kept (view heap slots made from it view its
    // host texture until the title rewrites them).
    if let Some(&r) = map.get(&(address, *desc))
        && current(r as This)
    {
        return Some(r as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let mut out = std::ptr::null();
    let made = create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    PLACED_BYTES.fetch_add(placed_size(desc).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    map.insert((address, *desc), out as usize);
    // Its host texture is made without the map's lock: the textures of the
    // slots a submission's shaders read are made several at a time.
    drop(map);
    resource(out).host().is_some().then_some(out)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
pub(crate) fn test_buffer(width: u64) -> This {
    let heap = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let desc = ResourceDesc {
        dimension: BUFFER,
        alignment: 0,
        width,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags: 0,
    };
    let mut out = std::ptr::null();
    let hr = create_committed(
        std::ptr::null(),
        &heap,
        0,
        &desc,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    assert_eq!(hr, S_OK);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_regions_rows_in_a_footprint() {
        // R32_FLOAT (41) 100x8x1, pitch 512.
        let f = PlacedFootprint { offset: 4096, format: 41, width: 100, height: 8, depth: 1, row_pitch: 512 };
        assert_eq!(f.rows([2, 3, 0], [10, 2, 1]), Some(vec![(3 * 512 + 8, 40), (4 * 512 + 8, 40)]));
        assert_eq!(f.rows([95, 0, 0], [10, 1, 1]), None, "past the width");
        // BC1 (71) 16x16, 8 bytes a 4x4 block, pitch 256: a block row per 4 texel rows.
        let bc = PlacedFootprint { offset: 0, format: 71, width: 16, height: 16, depth: 1, row_pitch: 256 };
        assert_eq!(bc.rows([4, 4, 0], [8, 8, 1]), Some(vec![(256 + 8, 16), (2 * 256 + 8, 16)]));
    }
    use crate::object::slot;

    #[test]
    fn a8_unorm_is_not_in_the_r8_typeless_family() {
        // DXGI_FORMAT values: 60 R8_TYPELESS, 61 R8_UNORM, 62 R8_UINT,
        // 63 R8_SNORM, 64 R8_SINT, 65 A8_UNORM. The R8 typeless family is
        // 60..=64; A8_UNORM has no typeless family and cannot be cast to R8,
        // so it must stand alone rather than fold into R8_TYPELESS.
        assert_eq!(family(60), 60);
        for f in 61..=64 {
            assert_eq!(family(f), 60);
        }
        assert_eq!(family(65), 65);
        // Boundaries: 59 belongs to the preceding family, 66 is unmapped.
        assert_eq!(family(59), 53);
        assert_eq!(family(66), 66);
    }

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    /// A title texture's host texture is made when first asked for, and a
    /// texture of the same memory made later with more uses (unordered
    /// access) shares it: the first one made takes every use the memory's
    /// title textures have.
    #[cfg(windows)]
    #[test]
    fn host_textures_are_made_on_first_use_with_every_use_of_their_memory() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let h = heap(1 << 20);
        let place = |desc: &ResourceDesc| {
            let mut out = null;
            assert_eq!(create_placed(null, h, 0, desc, 0, null.cast(), &RESOURCE_IIDS[0], &mut out), S_OK);
            out
        };
        let read = place(&texture(64, 64, 28, 1));
        let written = place(&ResourceDesc { flags: UNORDERED, ..texture(64, 64, 28, 1) });
        assert!(made(resource(read)).is_none() && made(resource(written)).is_none(), "made at creation");
        let first = host_copy(read).expect("a host texture");
        // SAFETY: a live resource.
        assert_ne!(unsafe { first.texture.GetDesc() }.Flags.0 as u32 & UNORDERED, 0, "the memory's uses");
        assert!(std::sync::Arc::ptr_eq(&first, &host_copy(written).unwrap()), "shared");
        for r in [read, written] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }
        assert_eq!(slot::<Release>(h, 0x10)(h), 0);
    }

    /// Host textures the GPU writes (render target, depth, unordered access)
    /// are never given up, and the title releasing its texture frees its
    /// host texture (nothing else holds it); one fed from title memory is
    /// given up and made again on its next use.
    #[cfg(windows)]
    #[test]
    fn only_textures_fed_from_title_memory_are_given_up() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let h = heap(1 << 20);
        let place = |offset: u64, desc: &ResourceDesc| {
            let mut out = null;
            assert_eq!(create_placed(null, h, offset, desc, 0, null.cast(), &RESOURCE_IIDS[0], &mut out), S_OK);
            out
        };
        let read = place(0, &texture(64, 64, 28, 1));
        let target = place(1 << 18, &ResourceDesc { flags: 1, ..texture(64, 64, 28, 1) });
        let (r, t) = (host_copy(read).unwrap(), host_copy(target).unwrap());
        let listed = candidates();
        assert!(listed.iter().any(|(c, _)| std::sync::Arc::ptr_eq(c, &r)), "fed from title memory");
        assert!(!listed.iter().any(|(c, _)| std::sync::Arc::ptr_eq(c, &t)), "the GPU writes it");
        let (weak_r, weak_t) = (std::sync::Arc::downgrade(&r), std::sync::Arc::downgrade(&t));
        drop((listed, r, t));
        let (r, t) = (weak_r, weak_t);
        assert!(!give_up_now(target), "never given up");
        assert!(give_up_now(read));
        // (Not checked freed: a pass in another test may hold it a moment.)
        assert!(made(resource(read)).is_none(), "given up");
        drop(r);
        assert!(host_copy(read).is_some(), "made again on its next use");
        assert_eq!(slot::<Release>(target, 0x10)(target), 0);
        assert!(t.upgrade().is_none(), "freed with its title texture");
        assert_eq!(slot::<Release>(read, 0x10)(read), 0);
        assert_eq!(slot::<Release>(h, 0x10)(h), 0);
    }

    /// Placed textures asked for from several threads at once: one resource
    /// and one host texture for each memory, made side by side.
    #[cfg(windows)]
    #[test]
    fn placed_textures_made_side_by_side_are_one_each() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let memory = vec![0u8; 4 << 16];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let base = memory.as_ptr() as u64;
        let made: Vec<Vec<usize>> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..4)
                .map(|_| s.spawn(move || (0..4u64).map(|k| texture_at(base + (k << 16), &words).unwrap() as usize).collect::<Vec<_>>()))
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        for k in 0..4 {
            assert!(made.iter().all(|m| m[k] == made[0][k]), "one resource for memory {k}");
            assert!(made_copy(made[0][k] as This).is_some(), "with a host texture");
        }
    }

    /// Textures idle for the idle time and frames are given up; past the
    /// budget the least recently used others follow until the excess is
    /// gone.
    #[test]
    fn idle_textures_go_and_past_the_budget_the_least_recently_used() {
        let (frame, now) = (1_000, 1_000_000);
        let recent = (frame, now - 1_000, 10);
        let idle = (frame - IDLE_FRAMES, now - IDLE_MS, 10);
        let older = (frame, now - 5_000, 30);
        let uses = [recent, idle, older];
        assert_eq!(choose(&uses, frame, now, 0), vec![1], "idle only");
        assert_eq!(choose(&uses, frame, now, 10), vec![1], "the idle one covers the excess");
        assert_eq!(choose(&uses, frame, now, 11), vec![1, 2], "then the least recently used");
        assert_eq!(choose(&uses, frame, now, 1 << 20), vec![1, 2, 0], "all of them if need be");
        let early = (frame - 1, now - IDLE_MS, 10);
        assert!(choose(&[early], frame, now, 0).is_empty(), "too few frames ago");
    }

    /// Title textures of one memory share a key in any format of the same
    /// element; another address, shape, element size or a depth texture is
    /// another memory. The library's own layout pick resolves as spelled.
    #[test]
    fn one_memory_one_key_in_any_format_of_its_element() {
        let a = ResourceDesc { flags: 4, ..texture(192, 108, 42, 1) };
        let b = ResourceDesc { format: 67, flags: 0, ..a };
        let key = memory_key(0x1_0000, &a);
        assert!(key.is_some());
        assert_eq!(key, memory_key(0x1_0000, &b));
        assert_eq!(key, memory_key(0x1_0000, &ResourceDesc { layout: 0x105, ..b }));
        assert_ne!(key, memory_key(0x2_0000, &a));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { width: 96, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { format: 10, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { layout: 0x102, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { flags: 2, format: 40, ..a }));
        assert_eq!(memory_key(0x1_0000, &buffer(256)), None);
    }

    /// Castable formats: every format of the same element but typeless and
    /// depth ones; block formats among block formats of their block size.
    #[test]
    fn castable_formats_share_the_element() {
        let c = castable(42);
        for f in [67, 28, 29, 41, 24, 26, 87, 35] {
            assert!(c.contains(&f), "{f}");
        }
        for f in [42, 39, 40, 45, 27, 10, 56] {
            assert!(!c.contains(&f), "{f}");
        }
        let bc1 = castable(71);
        assert!(bc1.contains(&72) && bc1.contains(&80) && bc1.contains(&81));
        assert!(!bc1.contains(&70) && !bc1.contains(&98) && !bc1.contains(&17));
        assert!(castable(98).contains(&95) && castable(98).contains(&77));
        assert!(castable(53).contains(&56) && !castable(53).contains(&55));
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as GTA VI asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    #[test]
    fn depth_stencil_footprints_follow_the_pc_plane_order() {
        // Native D3D12 reports the same layout for both depth/stencil families
        // and their view formats. Mips vary fastest, then array slices, then planes.
        let expected = [
            (512, 39, 16, 8),
            (4_608, 39, 8, 4),
            (6_656, 39, 16, 8),
            (10_752, 39, 8, 4),
            (12_800, 60, 16, 8),
            (16_896, 60, 8, 4),
            (18_944, 60, 16, 8),
            (23_040, 60, 8, 4),
        ];
        for format in (19..=22).chain(44..=47) {
            let desc = ResourceDesc {
                depth_or_array_size: 2,
                ..texture(16, 8, format, 2)
            };
            let mut layouts = [NO_FOOTPRINT; 8];
            let (mut rows, mut sizes, mut total) = ([0; 8], [0; 8], 0);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                0,
                8,
                512,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );

            assert_eq!(
                layouts.map(|l| (l.offset, l.format, l.width, l.height)),
                expected,
                "format {format}"
            );
            assert!(
                layouts.iter().all(|l| l.depth == 1 && l.row_pitch == 512),
                "format {format}"
            );
            assert_eq!(rows, [8, 4, 8, 4, 8, 4, 8, 4], "format {format}");
            assert_eq!(sizes, [64, 32, 64, 32, 16, 8, 16, 8], "format {format}");
            assert_eq!(total, 24_072, "format {format}");
            assert!(subresource(&desc, 8).is_none(), "format {format}");
        }
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                &host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    /// A texture region copied into a title buffer through our list and
    /// queue: once the list has run, its rows are in the buffer at the
    /// footprint's offset and pitch, and nothing else of the buffer moved.
    #[cfg(windows)]
    #[test]
    fn a_texture_read_back_reaches_the_title_buffer() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source, mut target) = (null, null, null);
        let created = |desc: &ResourceDesc, state, out: &mut This| {
            assert_eq!(create_committed(null, &PROPERTIES, 0, desc, state, null.cast(), iid, out), S_OK);
        };
        created(&texture(4, 4, 28, 1), COPY_DEST, &mut image);
        created(&buffer(1024), 0, &mut source);
        created(&buffer(4096), 0, &mut target);
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let (out, _) = buffer_bytes(target).unwrap();
        // SAFETY: the buffer holds 4096 bytes.
        unsafe { std::ptr::write_bytes(out, 0xEE, 4096) };
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &guid(0x0ec870a6, 0x5d7e, 0x4c22, [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed]), &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]), &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]), &mut list),
            S_OK
        );
        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint { offset: 0, format: 28, width: 4, height: 4, depth: 1, row_pitch: 256 };
        let image_at = Location { resource: image, kind: 0, footprint: rows };
        let from_source = Location { resource: source, kind: 1, footprint: rows };
        // Texels (1..3, 1..3) into the target's footprint at 512, at texel (2, 1).
        let into_target = Location { resource: target, kind: 1, footprint: PlacedFootprint { offset: 512, ..rows } };
        let region: [u32; 6] = [1, 1, 0, 3, 3, 1];
        type Copy = extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        let copy = slot::<Copy>(list, 0x80);
        copy(list, &image_at, 0, 0, 0, &from_source, null.cast());
        copy(list, &into_target, 2, 1, 0, &image_at, region.as_ptr().cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        // SAFETY: the buffer holds 4096 bytes; the submission has run.
        let read = unsafe { std::slice::from_raw_parts(out, 4096) };
        let mut want = vec![0xEEu8; 4096];
        for row in 0..2 {
            let (to, from) = (512 + (1 + row) * 256 + 2 * 4, (1 + row) * 256 + 4);
            want[to..to + 8].copy_from_slice(&bytes[from..from + 8]);
        }
        assert!(read == want.as_slice(), "the region's rows, and nothing else");
    }

    /// Two queues, as Grand Theft Auto VI's: a compute list copies a buffer,
    /// then waits for a value a graphics list writes at its end, then reads
    /// back the texture that graphics list fills. Submitted first, its first
    /// part runs at once and the rest waits for the graphics list, then
    /// reads what it wrote.
    #[cfg(windows)]
    #[test]
    fn a_list_waiting_for_another_queues_write_reads_what_it_wrote() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source, mut target, mut early, mut words) = (null, null, null, null, null);
        let created = |desc: &ResourceDesc, state, out: &mut This| {
            assert_eq!(create_committed(null, &PROPERTIES, 0, desc, state, null.cast(), iid, out), S_OK);
        };
        // A plain texture: its contents are the host's from the graphics
        // list's submission on (`texture::host_writes`), not from when the
        // compute list was recorded.
        created(&texture(4, 4, 28, 1), COPY_DEST, &mut image);
        created(&buffer(1024), 0, &mut source);
        created(&buffer(1024), 0, &mut target);
        created(&buffer(1024), 0, &mut early);
        created(&buffer(256), 0, &mut words);
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let (out, _) = buffer_bytes(target).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        unsafe { std::ptr::write_bytes(out, 0xEE, 1024) };
        let (flag, _) = buffer_bytes(words).unwrap();
        let flag = flag.cast::<u64>();

        let queue_iid = guid(0x0ec870a6, 0x5d7e, 0x4c22, [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed]);
        let allocator_iid = guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        let (mut graphics, mut compute, mut allocator, mut drawing, mut computing) = (null, null, null, null, null);
        for (kind, queue) in [(0u32, &mut graphics), (2, &mut compute)] {
            let desc = [kind, 0, 0, 0];
            assert_eq!(crate::queue::create(null, desc.as_ptr().cast(), &queue_iid, queue), S_OK);
        }
        assert_eq!(crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator), S_OK);
        for (kind, list) in [(0u32, &mut drawing), (2, &mut computing)] {
            assert_eq!(crate::list::create_list(null, 0, kind, allocator, null, &list_iid, list), S_OK);
        }
        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint { offset: 0, format: 28, width: 4, height: 4, depth: 1, row_pitch: 256 };
        let image_at = Location { resource: image, kind: 0, footprint: rows };
        type Copy = extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        type CopyBuffer = extern "system" fn(This, This, u64, This, u64, u64);
        type Close = extern "system" fn(This) -> HRESULT;
        type Execute = extern "system" fn(This, u32, *const This);
        const GREATER_EQUAL: u32 = 7;

        // Compute: [copy source → early][wait flag ≥ 1][image → target].
        slot::<CopyBuffer>(computing, 0x78)(computing, early, 0, source, 0, 1024);
        crate::list::wait64(computing, flag as u64, GREATER_EQUAL, 1, 0);
        let into_target = Location { resource: target, kind: 1, footprint: rows };
        slot::<Copy>(computing, 0x80)(computing, &into_target, 0, 0, 0, &image_at, null.cast());
        slot::<Close>(computing, 0x48)(computing);
        // Graphics: [source → image][flag = 1 at its end].
        let from_source = Location { resource: source, kind: 1, footprint: rows };
        slot::<Copy>(drawing, 0x80)(drawing, &image_at, 0, 0, 0, &from_source, null.cast());
        crate::list::write64(drawing, flag as u64, 1, 0, 0);
        slot::<Close>(drawing, 0x48)(drawing);

        slot::<Execute>(compute, 0x50)(compute, 1, &computing);
        let (copied, _) = buffer_bytes(early).unwrap();
        // SAFETY: the buffers hold 1024 bytes; nothing runs meanwhile.
        let read = |at: *mut u8| unsafe { std::slice::from_raw_parts(at, 1024) }.to_vec();
        assert_eq!(read(copied), bytes, "the part before the wait ran");
        assert!(read(out).iter().all(|&b| b == 0xEE), "the part after it waits");
        // SAFETY: the buffer holds 256 bytes.
        assert_eq!(unsafe { flag.read_volatile() }, 0);
        // D3D12 lets the title reset a list as soon as it has submitted it:
        // the held part keeps its own host list and what it reads.
        type Reset = extern "system" fn(This, This, This) -> HRESULT;
        assert_eq!(slot::<Reset>(computing, 0x50)(computing, allocator, null), S_OK);
        slot::<Close>(computing, 0x48)(computing);

        slot::<Execute>(graphics, 0x50)(graphics, 1, &drawing);
        // SAFETY: as above.
        assert_eq!(unsafe { flag.read_volatile() }, 1);
        let read = read(out);
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}: the texture as the graphics list filled it");
        }
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }
}
