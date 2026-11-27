//! Early depth (`emit`, "Early depth") for pixel programs the console
//! tests depth after (`synthetic_tests` has those it tests before): a
//! program that reads a texture view from the heap (so it carries read
//! marks) and exports a colour declares `[earlydepthstencil]`; one that
//! discards, exports depth, stencil or coverage, writes a UAV of the
//! title's or runs a primitive-ordered section does not. The loop limit's
//! discard does not count. These run with `VICTORY_SHADER_EARLY_Z` unset.

use super::container::SigElement;
use super::{lower, validator};
use crate::native::{HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// `exp mrt0 v2, v3, v4, v5 done vm`.
const EXP_MRT0: [u32; 2] = [0xF800_180F, 0x0504_0302];
/// `exp null off, off, off, off done vm`.
const EXP_NULL_DONE: [u32; 2] = [0xF800_1890, 0x0000_0000];
const POPS: u32 = 1 << 16;

/// The texel read: `s_mov_b32 s5, 0`, the descriptor at heap byte 32
/// (`s_load_dwordx8 s[12:19], s[4:5], 0x20`), `image_load v[2:5], v[0:1],
/// s[12:19] dmask:0xf dim:2D unorm` at (0, 0).
fn read() -> Vec<u32> {
    vec![
        0xBE85_0380,
        0xF40C_0302,
        0xFA00_0020,
        S_WAITCNT_ALL,
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_1F08,
        0x0003_0200,
        S_WAITCNT_ALL,
    ]
}

/// Reads, then exports the texel.
fn plain() -> Vec<u32> {
    let mut w = read();
    w.extend(EXP_MRT0);
    w.push(S_ENDPGM);
    w
}

/// Reads, then discards the lanes whose texel's red is above 0.
fn discarding() -> Vec<u32> {
    let mut w = read();
    w.extend([
        0xBE88_037E, // s_mov_b32 s8, exec_lo
        0x7C22_0480, // v_cmpx_lt_f32 exec, 0, v2 (the lanes killed)
        0x8A08_7E08, // s_andn2_b32 s8, s8, exec_lo
        0xBF84_0004, // s_cbranch_scc0 -> kill
        0xBEFE_0308, // s_mov_b32 exec_lo, s8
    ]);
    w.extend(EXP_MRT0);
    w.push(S_ENDPGM);
    w.extend(EXP_NULL_DONE); // kill:
    w.push(S_ENDPGM);
    w
}

/// Reads, writes the texel to a second view of the heap's (`image_store
/// v[2:5], v[0:1], s[20:27]`, the descriptor at heap byte 64), exports it.
fn storing() -> Vec<u32> {
    let mut w = read();
    w.extend([
        0xF40C_0502,
        0xFA00_0040,
        S_WAITCNT_ALL,
        0xF020_1F08,
        0x0005_0200,
    ]);
    w.extend(EXP_MRT0);
    w.push(S_ENDPGM);
    w
}

/// Reads, counts s0 to 4 in a loop whose back edge is a jump (which the
/// loop limit leaves through its abort block), exports.
fn looping() -> Vec<u32> {
    let mut w = read();
    w.extend([
        0xBE80_0380, // s_mov_b32 s0, 0
        0x8000_8100, // loop: s_add_u32 s0, s0, 1
        0xBF09_8400, // s_cmp_ge_u32 s0, 4
        0xBF85_0001, // s_cbranch_scc1 exit
        0xBF82_FFFC, // s_branch loop
    ]);
    w.extend(EXP_MRT0);
    w.push(S_ENDPGM);
    w
}

fn target(name: &str, kind: u8, comp: u8, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row: (kind == 16).then_some(0),
        col: 0,
        cols,
        usage: (1 << cols) - 1,
        half: false,
    }
}

fn colour() -> SigElement {
    target("SV_Target", 16, 9, 4)
}

/// `code` as a pixel program with the heap at s4, lowered with read marks
/// and early depth as `VICTORY_SHADER_EARLY_Z` says.
fn lower_only(
    code: &[u32],
    db_shader_control: u32,
    outputs: Vec<SigElement>,
) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: true,
    })
    .expect("lower")
}

/// [`lower_only`], its DXIL validated where dxil.dll is.
fn lowered(
    code: &[u32],
    db_shader_control: u32,
    outputs: Vec<SigElement>,
) -> lower::Lowered {
    let l = lower_only(code, db_shader_control, outputs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    l
}

#[test]
fn a_colour_only_program_with_read_marks_tests_depth_first() {
    let l = lowered(&plain(), 0, vec![colour()]);
    assert!(l.view_marks && l.heap, "the read is marked");
    assert!(l.early_depth);
    // The console's own late-Z order changes nothing without side effects.
    assert!(lowered(&plain(), 0x10, vec![colour()]).early_depth);
}

#[test]
fn a_discard_keeps_depth_late() {
    let l = lowered(&discarding(), 0, vec![colour()]);
    assert!(l.discards > 0);
    assert!(!l.early_depth);
}

#[test]
fn the_loop_limits_discard_does_not_count() {
    let l = lowered(&looping(), 0, vec![colour()]);
    assert_eq!(l.discards, 1, "the loop limit's abort");
    assert!(l.early_depth);
}

#[test]
fn the_titles_own_writes_keep_depth_late() {
    let l = lowered(&storing(), 0, vec![colour()]);
    assert!(l.heap_writes, "a heap UAV written");
    assert!(!l.early_depth);
}

#[test]
fn an_ordered_section_keeps_depth_late() {
    assert!(!lower_only(&plain(), POPS, vec![colour()]).early_depth);
}

#[test]
fn depth_stencil_and_coverage_outputs_keep_depth_late() {
    // exp mrtz v2 (not done), then the colour.
    let mut w = read();
    w.extend([0xF800_1081, 0x0000_0002]);
    w.extend(EXP_MRT0);
    w.push(S_ENDPGM);
    let depth = target("SV_Depth", 17, 9, 1);
    assert!(
        !lowered(&w, 1, vec![colour(), depth]).early_depth,
        "depth export"
    );
    // Outputs the signature declares: depth, conservative depth, stencil
    // reference, coverage.
    for kind in [17, 18, 19, 20, 14] {
        let other = target(
            "SV_Other",
            kind,
            if matches!(kind, 17..=19) { 9 } else { 5 },
            1,
        );
        assert!(
            !lower_only(&plain(), 0, vec![colour(), other]).early_depth,
            "kind {kind}"
        );
    }
    // Or DB_SHADER_CONTROL enables: Z (bit 0), stencil test and op values
    // (1, 2), coverage to mask (7), mask (8).
    for bit in [0, 1, 2, 7, 8] {
        assert!(
            !lower_only(&plain(), 1 << bit, vec![colour()]).early_depth,
            "bit {bit}"
        );
    }
}
