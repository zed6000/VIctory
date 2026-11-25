//! Control flow and lanes, differentially: HLSL shapes that exercise the
//! structurisation (nested branches with early returns, loops with
//! divergent exits and values read after them, `break`/`continue`, a
//! switch, short-circuit conditions, branches on wave-wide tests, discard
//! and derivatives after branches), compiled by the console's compiler and
//! translated, against the PC compiler's, both run on the hardware adapter
//! and on WARP with per-lane-varying inputs. The console executes its own
//! code faithfully, so for integer and control-flow results the HLSL
//! semantics are the reference: results must be equal bit for bit.
//!
//! Needs `VICTORY_CONSOLE_DXC` and a PC `dxc.exe` (as `diff_tests`);
//! skipped without them.

use std::path::{Path, PathBuf};

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Draw, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, RootType, translate, validator};

fn tools() -> Option<(PathBuf, PathBuf)> {
    let console = PathBuf::from(std::env::var_os("VICTORY_CONSOLE_DXC")?);
    let pc = match std::env::var_os("VICTORY_PC_DXC") {
        Some(p) => PathBuf::from(p),
        None => validator::Validator::default_path()?.with_file_name("dxc.exe"),
    };
    (console.exists() && pc.exists()).then_some((console, pc))
}

fn compile(
    dxc: &Path,
    dir: &Path,
    text: &str,
    entry: &str,
    profile: &str,
) -> Result<Vec<u8>, String> {
    let input = dir.join(format!("{entry}.hlsl"));
    let output = dir.join(format!("{entry}.bin"));
    std::fs::write(&input, text).map_err(|e| e.to_string())?;
    let out = std::process::Command::new(dxc)
        .args(["-T", profile, "-E", entry, "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    std::fs::read(output).map_err(|e| e.to_string())
}

const CS_COMMON: &str = r#"
#define RS "DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0))"
ByteAddressBuffer inb : register(t0);
RWByteAddressBuffer outb : register(u0);
#define CS [RootSignature(RS)] [numthreads(64, 1, 1)] void cs(uint3 d : SV_DispatchThreadID, uint li : SV_GroupIndex)
#define IN(K_) inb.Load(16 * d.x + 4 * (K_))
#define OUT(K_, X_) outb.Store(16 * d.x + 4 * (K_), (X_))
"#;

/// Compute shapes. Each lane reads four dwords (small varied integers in
/// the low bits, so branches diverge within every wave) and writes four.
const CS_CASES: &[(&str, &str)] = &[
    (
        "nested branches with early returns",
        r#"CS {
            uint a = IN(0), b = IN(1), c = IN(2);
            OUT(3, 7);
            if (a & 1) {
                if (b & 2) { OUT(0, a + b); OUT(1, 1); return; }
                if (c & 4) { OUT(0, a * 3); }
                else { OUT(0, b ^ c); OUT(2, 9); return; }
            } else if ((a & 6) == 4) {
                OUT(0, c << 2); OUT(1, 2); return;
            }
            OUT(1, a | b);
            OUT(2, a > b ? a - b : b - a);
        }"#,
    ),
    // A value the loop defines, read after it: each lane's own last value,
    // not the wave's last iteration (TODO: "values a divergent loop defines
    // and code reads after it").
    (
        "divergent loop exit and values read after it",
        r#"CS {
            uint a = IN(0), b = IN(1), n = IN(2) & 15;
            uint last = 1000, s = 0, i;
            for (i = 0; i < n; i++) {
                s += a ^ i;
                last = s * 3 + i;
                if (((s + b) & 7) == 5) break;
            }
            OUT(0, s); OUT(1, last); OUT(2, i); OUT(3, n);
        }"#,
    ),
    (
        "break and continue",
        r#"CS {
            uint a = IN(0), b = IN(1), n = IN(2) & 31, s = 0, odd = 0;
            [loop] for (uint i = 0; i < n; i++) {
                if ((i ^ a) & 1) { odd++; continue; }
                s += i * (b & 7);
                if (s > (a & 63)) break;
                s ^= 1;
            }
            OUT(0, s); OUT(1, odd); OUT(2, a); OUT(3, b);
        }"#,
    ),
    (
        "switch",
        r#"CS {
            uint a = IN(0), b = IN(1), r = 0;
            [branch] switch (a & 7) {
                case 0: r = b; break;
                case 1: r = b * 2;
                case 2: r += 5; break;
                case 5: r = b ^ 0xFF; break;
                case 6: r = IN(2); break;
                default: r = 99;
            }
            OUT(0, r); OUT(1, a & 7); OUT(2, b); OUT(3, r + 1);
        }"#,
    ),
    (
        "short-circuit conditions",
        r#"CS {
            uint a = IN(0), b = IN(1), c = IN(2);
            uint r = 0;
            if ((a & 1) && (b & 2) || (c & 4) && !(a & 8)) r = 1;
            if ((a & 3) == 3 || ((b & 3) == 0 && inb.Load(4 * (c & 63)) > a)) r += 2;
            bool t = (a > b) && (b > c);
            OUT(0, r); OUT(1, t ? 5 : 6); OUT(2, (a & 1) || (c & 1)); OUT(3, 0);
        }"#,
    ),
    (
        "nested loops with divergent trip counts",
        r#"CS {
            uint a = IN(0), b = IN(1), s = 0, t = 0;
            for (uint i = 0; i < (a & 7); i++) {
                for (uint j = 0; j < (b & 7); j++) {
                    if (((i + j) & 3) == 3) continue;
                    s += i * j + 1;
                    if (s > 40) break;
                }
                t += s;
                if ((t & 15) == 9) break;
            }
            OUT(0, s); OUT(1, t); OUT(2, a & 7); OUT(3, b & 7);
        }"#,
    ),
    (
        "values joined from both sides",
        r#"CS {
            uint a = IN(0), b = IN(1), c = IN(2);
            uint x, y = c;
            if (a & 1) { x = a * 5; y = b; }
            else if (a & 2) { x = b + 1; }
            else { x = c ^ 0xAA; y = a; }
            uint z = (b & 4) ? x : y;
            OUT(0, x); OUT(1, y); OUT(2, z); OUT(3, x ^ y);
        }"#,
    ),
    // Branches on wave-wide tests over per-lane conditions: per-lane code
    // inside a wave-uniform branch (TODO: "per-lane branches for
    // wave-wide mask tests").
    (
        "branches on wave-wide tests",
        r#"CS {
            uint a = IN(0), b = IN(1), r = 0;
            if (WaveActiveAnyTrue(a == 77)) r = 1;
            if (WaveActiveAllTrue(b < 1000)) r |= 2;
            if (WaveActiveAnyTrue((a & 3) == 1)) { if (b & 1) r |= 4; }
            uint lane = WaveGetLaneIndex();
            if (a & 1) r |= WaveActiveCountBits((b & 2) != 0) << 8;
            OUT(0, r); OUT(1, a); OUT(2, b); OUT(3, lane < 64);
        }"#,
    ),
];

const PS_COMMON: &str = r#"
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=4, b0)"
cbuffer K : register(b0) { float4 k; };
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)] V vs(float4 p : POSITION, float2 uv : TEXCOORD0) { V o; o.pos = p; o.uv = uv; return o; }
"#;

/// Pixel shapes, drawn over 16x16 targets (results are small integers and
/// exact fractions).
const PS_CASES: &[(&str, &str)] = &[
    (
        "discard in nested branches",
        r#"[RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float r = 0;
            if (p.x & 1) {
                if ((p.y & 3) == 2) discard;
                r = 1;
            } else {
                [loop] for (uint n = 0; n < (p.y & 7); n++) {
                    r += 0.25;
                    if (n == (p.x & 3)) { if (p.y & 8) discard; break; }
                }
            }
            return float4(r, p.x, p.y, 1);
        }"#,
    ),
    // Derivatives of a value after divergent branches joined (quad-uniform
    // derivatives of per-pixel values computed on either side).
    (
        "derivatives after branches",
        r#"[RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float v;
            if ((p.x ^ p.y) & 2) v = p.x * 2.0 + p.y;
            else v = p.y * 4.0 - p.x;
            if (p.y & 4) v += 8;
            return float4(ddx_fine(v), ddy_fine(v), v, 1);
        }"#,
    ),
    // A discarded pixel stays a helper on the console (its code re-enables
    // whole quads), so the survivors' derivatives still read it.
    (
        "discard then derivatives of the survivors",
        r#"[RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float v = p.x * 3.0 + p.y * 5.0;
            if (((p.x * 7 + p.y * 3) & 5) == 5) discard;
            return float4(ddx_coarse(v), ddy_coarse(v), v, 1);
        }"#,
    ),
    (
        "two discards, then derivatives",
        r#"[RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float v = p.x * 3.0 + p.y * 5.0;
            if ((p.x & 3) == 1 && (p.y & 1)) discard;
            float w = v * v;
            if (((p.x + p.y) & 7) == 6) discard;
            return float4(ddx_fine(v), ddy_fine(w), ddx_coarse(w), 1);
        }"#,
    ),
    (
        "discard in a loop, then derivatives",
        r#"[RootSignature(RS)] float4 ps(V i) : SV_Target {
            uint2 p = uint2(i.pos.xy);
            float v = 0;
            [loop] for (uint n = 0; n < (p.y & 3) + 1; n++) {
                v += p.x * (n + 1.0);
                if (((p.x ^ n) & 7) == 5) discard;
            }
            return float4(ddx_fine(v), ddy_fine(v), v, 1);
        }"#,
    ),
    (
        "discard with depth and a second target",
        r#"struct O { float4 c : SV_Target0; float4 d : SV_Target1; float z : SV_Depth; };
        [RootSignature(RS)] O ps(V i) {
            uint2 p = uint2(i.pos.xy);
            float v = p.x * 2.0 + p.y;
            if (((p.x * 5 + p.y) & 3) == 2) discard;
            O o;
            o.c = float4(ddx_coarse(v), ddy_fine(v), v, 1);
            o.d = float4(v, 1, 2, 3);
            o.z = 0.25;
            return o;
        }"#,
    ),
];

/// Per-lane inputs: four dwords per lane from a fixed hash (low bits vary
/// lane to lane; a few lanes hold larger values).
fn lane_inputs(lanes: u32) -> Vec<u32> {
    (0..lanes * 4)
        .map(|n| {
            let h = n.wrapping_mul(0x9E37_79B9) ^ n >> 3;
            if n % 11 == 0 { h } else { h % 97 }
        })
        .collect()
}

fn console_user_data(s: &HostShader) -> Vec<u32> {
    let mut data = vec![0; (s.user_data_dwords() as usize).max(1)];
    for u in &s.user_data {
        data[u.index as usize] = match (u.root, u.slot) {
            (RootType::Table, 0) => 0,
            (RootType::Table, 1) => 32,
            (RootType::Constants, 0) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    data
}

#[test]
fn control_flow_runs_as_pc_compiles_it() {
    let failures = run(&|_| true);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The wave-wide case's results computed from the inputs for waves of
/// `wave` lanes (the console program's).
fn wave_model(data: &[u32], wave: usize) -> Vec<u32> {
    let lanes = data.len() / 4;
    let mut out = vec![0u32; data.len()];
    for w in (0..lanes).step_by(wave) {
        let ls = w..(w + wave).min(lanes);
        let a = |l: usize| data[4 * l];
        let b = |l: usize| data[4 * l + 1];
        let any77 = ls.clone().any(|l| a(l) == 77);
        let all_small = ls.clone().all(|l| b(l) < 1000);
        let any1 = ls.clone().any(|l| a(l) & 3 == 1);
        let bits = ls
            .clone()
            .filter(|&l| a(l) & 1 != 0 && b(l) & 2 != 0)
            .count() as u32;
        for l in ls {
            let mut r = u32::from(any77) | u32::from(all_small) << 1;
            if any1 && b(l) & 1 != 0 {
                r |= 4;
            }
            if a(l) & 1 != 0 {
                r |= bits << 8;
            }
            out[4 * l..4 * l + 4].copy_from_slice(&[r, a(l), b(l), 1]);
        }
    }
    out
}

/// The wave size of a console container's program.
fn console_wave(container: &[u8]) -> usize {
    let c = crate::dxbc::Container::parse(container).expect("a container");
    let part = c.part(crate::dxbc::FourCC::DXIL).expect("a DXIL part");
    let programs = crate::native_programs(part.data).expect("native code");
    if programs[0].wave64 { 64 } else { 32 }
}

/// Runs the cases `pick` names on both adapters; returns the failures.
fn run(pick: &dyn Fn(&str) -> bool) -> Vec<String> {
    let Some((console, pc)) = tools() else {
        eprintln!("no console or PC compiler; skipped");
        return Vec::new();
    };
    let dir = std::env::temp_dir().join(format!("victory-flow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lanes = 256;
    let data = lane_inputs(lanes);
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(lanes * 4),
    ];
    let mut failures = Vec::new();
    let only = std::env::var("VICTORY_DIFF_ONLY").ok();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (name, src) in CS_CASES
            .iter()
            .filter(|c| only.as_ref().is_none_or(|o| o == c.0) && pick(c.0))
        {
            let text = format!("{CS_COMMON}\n{src}\n");
            let mut wave = 64;
            let ours = match compile(&console, &dir, &text, "cs", "cs_6_5").and_then(|c| {
                wave = console_wave(&c);
                translate(&c).map_err(|e| e.to_string())
            }) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{name}: {e}");
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let (root_t, pso_t) = match host.compute_pipeline(&ours) {
                Ok(p) => p,
                // WARP runs no fixed wave size.
                Err(e) if warp => {
                    eprintln!("{name}: skipped on WARP ({e})");
                    continue;
                }
                Err(e) => panic!("{name}: {e}"),
            };
            let dispatch = |roots| Dispatch {
                roots,
                views: &views,
                samplers: &[],
                groups: [lanes / 64, 1, 1],
            };
            let got = host
                .dispatch(
                    &root_t,
                    &pso_t,
                    &dispatch(vec![(0, Root::Constants(console_user_data(&ours)))]),
                )
                .unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
            let pc_cs = compile(&pc, &dir, &text, "cs", "cs_6_5").unwrap();
            // SAFETY: the PC container carries its root signature.
            let root_p: ID3D12RootSignature =
                unsafe { host.device.CreateRootSignature(0, &pc_cs) }.unwrap();
            let pso_p = host.compute_pipeline_with(&root_p, &pc_cs).unwrap();
            let table = |index| Root::Table {
                sampler: false,
                index,
            };
            let mut want = host
                .dispatch(
                    &root_p,
                    &pso_p,
                    &dispatch(vec![(0, table(0)), (1, table(1))]),
                )
                .unwrap();
            // Wave-wide tests depend on the wave size: the console's, from
            // the inputs, not the PC compile's.
            if name.contains("wave-wide") {
                want[0] = wave_model(&data, wave);
            }
            let bad: Vec<usize> = (0..got[0].len())
                .filter(|&w| got[0][w] != want[0][w])
                .collect();
            eprintln!(
                "{name}: {label}: {} of {} words differ",
                bad.len(),
                got[0].len()
            );
            if let Some(&w) = bad.first() {
                failures.push(format!(
                    "{name}: {label}: lane {} word {}: {:#x} vs {:#x} (in {:x?}); {} words differ",
                    w / 4,
                    w % 4,
                    got[0][w],
                    want[0][w],
                    &data[w / 4 * 4..w / 4 * 4 + 4],
                    bad.len()
                ));
            }
        }
        // Pixel shapes.
        let vertices = [
            -1.0, -1.0, 0.5, 1.0, 0.0, 1.0, //
            -1.0, 3.0, 0.5, 1.0, 0.0, -1.0, //
            3.0, -1.0, 0.5, 1.0, 2.0, 1.0,
        ];
        let draw = |roots| Draw {
            roots,
            textures: &[],
            samplers: &[],
            vertices: &vertices,
            stride: 24,
            size: (16, 16),
            mesh_groups: 0,
        };
        for (name, src) in PS_CASES
            .iter()
            .filter(|c| only.as_ref().is_none_or(|o| o == c.0) && pick(c.0))
        {
            let text = format!("{PS_COMMON}\n{src}\n");
            let shaders = compile(&console, &dir, &text, "vs", "vs_6_5").and_then(|vs| {
                let ps = compile(&console, &dir, &text, "ps", "ps_6_5")?;
                Ok((
                    translate(&vs).map_err(|e| e.to_string())?,
                    translate(&ps).map_err(|e| e.to_string())?,
                ))
            });
            let (vs_c, ps_c) = match shaders {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{name}: {e}");
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let (root_t, pso_t) = host.pipeline(&vs_c, &ps_c).unwrap();
            let got = host
                .draw(
                    &root_t,
                    &pso_t,
                    &draw(vec![
                        (0, Root::Constants(console_user_data(&vs_c))),
                        (1, Root::Constants(console_user_data(&ps_c))),
                    ]),
                )
                .unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
            let (vs_p, ps_p) = (
                compile(&pc, &dir, &text, "vs", "vs_6_5").unwrap(),
                compile(&pc, &dir, &text, "ps", "ps_6_5").unwrap(),
            );
            // SAFETY: the PC container carries its root signature.
            let root_p: ID3D12RootSignature =
                unsafe { host.device.CreateRootSignature(0, &ps_p) }.unwrap();
            let pso_p = host.pipeline_with(&root_p, &vs_p, &ps_p).unwrap();
            let want = host
                .draw(
                    &root_p,
                    &pso_p,
                    &draw(vec![(0, Root::Constants(vec![0; 4]))]),
                )
                .unwrap();
            let bad: Vec<usize> = (0..got.len()).filter(|&t| got[t] != want[t]).collect();
            eprintln!(
                "{name}: {label}: {} of {} texels differ",
                bad.len(),
                got.len()
            );
            if let Some(&t) = bad.first() {
                failures.push(format!(
                    "{name}: {label}: texel ({}, {}): {:?} vs {:?}; {} texels differ",
                    t % 16,
                    t / 16,
                    got[t],
                    want[t],
                    bad.len()
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    failures
}
