//! Lane-mask phi lowering and console-lane dependency scans (split from
//! [`super`] to keep `emit/lower.rs` within the asset-policy file size
//! limit).

use std::collections::{HashMap, HashSet};

use super::{Lower, MaskData, Stop, Val, Why, unsupported};
use crate::emit::bitcode::V;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Whether any branch condition, instruction guard, or vector select/carry-in
/// mask reaches an opaque or constant lane set (including transitively across
/// `phi_masks`), requiring the console wave size and lane order.
pub(super) fn mask_uses_console_lanes(
    p: &Program,
    st: &Structured,
    phi_at: &HashMap<Value, (usize, usize)>,
) -> bool {
    let whens = st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let selects = p
        .blocks
        .iter()
        .enumerate()
        .filter(|&(b, _)| st.reach[b] != Bool::FALSE)
        .flat_map(|(_, block)| &block.insts)
        .filter_map(|i| {
            let Fields::Valu(v) = &i.fields else {
                return None;
            };
            if !matches!(
                i.name(),
                "v_cndmask_b32"
                    | "v_add_co_ci_u32"
                    | "v_sub_co_ci_u32"
                    | "v_subrev_co_ci_u32"
            ) {
                return None;
            }
            let &Operand::Scalar { code, .. } = v.src.get(2)? else {
                return None;
            };
            if code == NULL {
                return None;
            }
            let m = i.read(Reg::S(code))?;
            st.masks.get(&m).copied()
        });
    let mut seen_bools = HashSet::new();
    let mut seen_phis = HashSet::new();
    let mut work: Vec<Bool> = st
        .guards
        .iter()
        .flatten()
        .flatten()
        .copied()
        .chain(whens)
        .chain(selects)
        .collect();
    while let Some(g) = work.pop() {
        if !seen_bools.insert(g) {
            continue;
        }
        for a in st.bools.support(g) {
            match a {
                Atom::Opaque(_) | Atom::Constant(_) => return true,
                Atom::Phi(id) if seen_phis.insert(id) => {
                    if let Some(edges) = st.phi_masks.get(&id)
                        && let Some(&(b, i)) = phi_at.get(&id)
                    {
                        let phi = &p.blocks[b].phis[i];
                        let uniform_join = p.blocks[b].preds.iter().all(|&q| match st.exits[q] {
                            LaneExit::Branch {
                                when,
                                taken,
                                not_taken,
                            } if taken != not_taken => st
                                .bools
                                .support(when)
                                .iter()
                                .all(|a| matches!(a, Atom::Uniform(_))),
                            _ => true,
                        });
                        for (n, &e) in edges.iter().enumerate() {
                            if uniform_join || st.masks.contains_key(&phi.args[n]) {
                                work.push(e);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    false
}

impl Lower<'_> {
    fn fill_mask_phis(&mut self) -> Result<(), Stop> {
        const UNTRACKED: Why = "a lane-mask join of an untracked mask";
        let p = self.x.p;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let uniform = self.uniform_join(b);
            let forward_uniform = uniform
                || p.blocks[b]
                    .preds
                    .iter()
                    .filter(|&&q| self.block[q].is_some() && self.rpo[q] < self.rpo[b])
                    .all(|&q| match self.x.st.exits[q] {
                        LaneExit::Branch {
                            when,
                            taken,
                            not_taken,
                        } if taken != not_taken => self
                            .x
                            .st
                            .bools
                            .support(when)
                            .iter()
                            .all(|a| matches!(a, Atom::Uniform(_))),
                        _ => true,
                    });
            let number = self.console_lanes && !self.x.native.wave64 && uniform;
            let phi_edges = self.x.st.phi_masks.get(&phi.value);
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                let phi_edge = phi_edges.and_then(|edges| edges.get(n).copied());
                let x = match (self.x.st.masks.get(&arg), phi_edge, self.vals.get(&arg)) {
                    (Some(&m), _, _) => self
                        .boolean(phi_edge.unwrap_or(m), qb)
                        .map_err(|w| unsupported(offset, w))?,
                    (None, Some(m), _) if uniform => {
                        self.boolean(m, qb).map_err(|w| unsupported(offset, w))?
                    }
                    (None, _, Some(&Val::I32(word))) if number => self.lane_bit(qb, word, None),
                    (None, _, _) if !uniform && self.no_mask_lane_on_edge(b, &phi.args, n) => {
                        self.s.m.bool(false)
                    }
                    (None, Some(m @ (Bool::FALSE | Bool::TRUE)), _)
                        if forward_uniform && self.rpo[q] < self.rpo[b] =>
                    {
                        self.boolean(m, qb).map_err(|w| unsupported(offset, w))?
                    }
                    (None, _, _) => return Err(unsupported(offset, UNTRACKED)),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        Ok(())
    }

    pub(super) fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        let data_phis = std::mem::take(&mut self.data_phis);
        let mut resolved: Vec<Vec<Option<V>>> = data_phis
            .iter()
            .map(|&(_, b, _)| vec![None; p.blocks[b].preds.len()])
            .collect();
        loop {
            self.fill_mask_phis()?;
            let prev_read = self.read_phis.len();
            let live = self.live_edges();
            let mut progress = false;
            for (idx, &(_, b, i)) in data_phis.iter().enumerate() {
                let phi = &p.blocks[b].phis[i];
                let bit = phi.reg == Reg::Scc;
                for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                    if self.tail[q].is_none()
                        || resolved[idx][n].is_some()
                        || !live.contains(&(phi.value, n))
                    {
                        continue;
                    }
                    let arg = phi.args[n];
                    let x = match (self.vals.get(&arg).copied(), bit) {
                        (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                        // No lane from this edge reads it, or no lane comes
                        // along it at all: any value will do.
                        _ if self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg) => {
                            if bit {
                                self.s.m.bool(false)
                            } else {
                                self.s.m.i32(0)
                            }
                        }
                        // An input an unguarded counted `v_movreld` loop overwrites
                        // before anything reads it (PR #457).
                        (Some(Val::Missing("barycentrics read directly")), false)
                            if self.overwritten_movreld_edge(b, phi, n).is_some() =>
                        {
                            self.s.m.undef(self.s.t.i32)
                        }
                        (Some(Val::Bit(_)), false) => {
                            return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                        }
                        // A lane mask joined with data at a join every lane
                        // enters along one edge: the register holds the mask's
                        // number there (console lanes, so exact).
                        _ if !bit
                            && self.mask_data == MaskData::Console
                            && (self.x.st.masks.contains_key(&arg)
                                || self.x.st.highs.contains_key(&arg))
                            && self.uniform_join(b) =>
                        {
                            self.mask_dword(arg)
                                .map_err(|w| Stop::Retry(phi.value, w))?
                        }
                        (Some(Val::Missing(why)), _) => {
                            return Err(Stop::Retry(phi.value, why));
                        }
                        _ => {
                            return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                        }
                    };
                    resolved[idx][n] = Some(x);
                    progress = true;
                }
            }
            if self.pending_masks.is_empty() && self.read_phis.len() == prev_read && !progress {
                break;
            }
        }
        for (idx, (v, b, i)) in data_phis.into_iter().enumerate() {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = match resolved[idx][n] {
                    Some(x) => x,
                    None => match (self.vals.get(&phi.args[n]).copied(), bit) {
                        (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                        _ if bit => self.s.m.bool(false),
                        _ => self.s.m.i32(0),
                    },
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Lowered, Parts, lower};
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Lowers a compute program of constants or wave64 params: whether it translates.
    fn lowers_cs(
        words: &[u32],
        wave64: bool,
        const_sgprs: u8,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let layout = if const_sgprs > 0 {
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: u32::from(const_sgprs),
                base_register: 0,
                user_data: 0,
                slot: 0,
            }]
        } else {
            vec![]
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: u32::from(const_sgprs) << 1,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
        })
    }

    fn validate_dxil(dxil: &[u8]) {
        #[cfg(windows)]
        {
            use crate::emit::validator::Validator;
            if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
                let mut copy = dxil.to_vec();
                let ours = copy[4..20].to_vec();
                v.validate(&mut copy).expect("DXIL validation failed");
                assert_eq!(copy[4..20], ours[..], "container hash mismatch");
            }
        }
        #[cfg(not(windows))]
        let _ = dxil;
    }

    #[test]
    fn lane_mask_phi_joining_zero_and_compare_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_0380, // 8: s_mov_b32 s2, 0
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 1).unwrap();
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn wave64_loop_mask_accumulator_from_zero_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s[2:3]
        let words = [
            0xBE82_0480, // 0: s_mov_b64 s[2:3], 0
            0xBE84_0380, // 4: s_mov_b32 s4, 0
            0xD501_0001,
            cnd,         // 8: v_cndmask_b32 v1, 1, 2, s[2:3]
            0x7D84_0080, // 16: v_cmp_eq_u32 vcc, 0, v0
            0x8882_026A, // 20: s_or_b64 s[2:3], vcc, s[2:3]
            0x8004_8104, // 24: s_add_u32 s4, s4, 1
            0xBF0A_8404, // 28: s_cmp_lt_u32 s4, 4
            0xBF85_FFF9, // 32: s_cbranch_scc1 -> 8
            0xD501_0004,
            cnd, // 36: v_cndmask_b32 v4, 1, 2, s[2:3]
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, true, 0).unwrap();
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn lane_mask_phi_with_partial_constant_enables_console_lanes() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_038F, // 8: s_mov_b32 s2, 15 (partial constant mask 0xF)
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 1).unwrap();
        assert!(lowered.wave_ops, "expected WAVE_OPS for console_lanes");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn wave64_lane_mask_phi_with_partial_constant_validates() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s[2:3]
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_048F, // 8: s_mov_b64 s[2:3], 15 (partial constant mask 0xF)
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s[2:3], 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s[2:3]
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, true, 1).unwrap();
        assert!(lowered.wave_ops, "expected WAVE_OPS for Wave64 console_lanes");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn data_join_materializes_a_loop_mask_validates() {
        let words = [
            0xBE82_0380, // 0: s_mov_b32 s2, 0
            0xBE84_0380, // 4: s_mov_b32 s4, 0
            0x7D84_0004, // 8: v_cmp_eq_u32 vcc_lo, s4, v0
            0x8902_026A, // 12: s_xor_b32 s2, vcc_lo, s2
            0x8004_8104, // 16: s_add_u32 s4, s4, 1
            0xBF0A_8404, // 20: s_cmp_lt_u32 s4, 4
            0xBF85_FFFB, // 24: s_cbranch_scc1 -> 8
            0xBF06_8000, // 28: s_cmp_eq_u32 s0, 0
            0xBF85_0003, // 32: s_cbranch_scc1 -> 48
            0xBE86_03FF,
            0x0BAD_F00D, // 36: s_mov_b32 s6, 0x0BAD_F00D
            0xBF82_0001, // 44: s_branch -> 52
            0x8706_6A02, // 48: s_and_b32 s6, s2, vcc_lo
            0x7E02_0206, // 52: v_mov_b32 v1, s6
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let s2_phi = p
            .blocks
            .iter()
            .flat_map(|b| &b.phis)
            .find(|phi| phi.reg == crate::ir::Reg::S(2))
            .expect("loop mask phi for s2");
        assert!(
            st.phi_masks.contains_key(&s2_phi.value),
            "expected loop s2 phi to remain an Atom::Phi in phi_masks"
        );
        let lowered = lowers_cs(&words, false, 1).expect("lower");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn compare_mask_joined_with_data_on_skip_edge_lowers_and_refuses_saved_exec() {
        let mut words = [
            0xBE86_0385, // s_mov_b32 s6, 5
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0xBF88_0002, // s_cbranch_execz +2
            0xD4C4_0006,
            0x0002_0088, // v_cmp_gt_u32 s6, 8, v0
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
            0xD501_0003,
            0x0019_0280, // v_cndmask_b32 v3, 0, 1, s6
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 0).expect("lower");
        assert!(!lowered.wave_ops);
        validate_dxil(&lowered.dxil);
        words[4..6].copy_from_slice(&[0xBE86_0308, 0xBF80_0000]); // s_mov_b32 s6, s8; s_nop 0
        let Err(crate::ShaderError::Unsupported { what, .. }) = lowers_cs(&words, false, 0) else {
            panic!("a mask holding lanes off its edge joined with data translated");
        };
        assert!(what.contains("lane-mask join"), "{what}");
    }

    #[test]
    fn uniform_loop_mask_joined_with_data_validates() {
        let words = [
            0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
            0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
            0xBE87_0380, // s_mov_b32 s7, 0
            0xD501_0004,
            0x0019_0280, // header: v_cndmask_b32 v4, 0, 1, s6
            0x9006_8406, // s_lshr_b32 s6, s6, 4
            0x7E0A_0206, // v_mov_b32 v5, s6
            0x7E0C_0505, // v_readfirstlane_b32 s6, v5
            0x8007_8107, // s_add_u32 s7, s7, 1
            0xBF0A_8207, // s_cmp_lt_u32 s7, 2
            0xBF85_FFF8, // s_cbranch_scc1 header
            0x7E06_0206, // v_mov_b32 v3, s6
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 0).expect("lower");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn per_lane_mask_phi_read_as_whole_mask_and_readlane_validates() {
        let words = [
            0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0 (v0 < s0)
            0xBE82_0380, // s_mov_b32 s2, 0
            0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
            0xBF88_0001, // s_cbranch_execz -> restore
            0xBE82_036A, // s_mov_b32 s2, vcc_lo
            0xBEFE_0304, // restore: s_mov_b32 exec_lo, s4
            0x8706_7E02, // s_and_b32 s6, s2, exec_lo
            0xD501_0002,
            (0x80) | (0x81 << 9) | (2 << 18), // v_cndmask_b32 v2, 0, 1, s2
            0xD760_0007,
            0x0001_0102, // v_readlane_b32 s7, v2, 0
            0xD760_0008,
            0x0001_3F02, // v_readlane_b32 s8, v2, 31
            0x7E06_0206, // v_mov_b32 v3, s6
            0x7E08_0207, // v_mov_b32 v4, s7
            0x7E0A_0208, // v_mov_b32 v5, s8
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 1).expect("lower");
        assert!(lowered.wave_ops);
        assert_eq!(lowered.wave_ballots, 1);
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn fill_phis_fixpoint_resolves_loop_mask_ballot_without_empty_phi_or_duplicate_ballots() {
        let words = [
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8C_0380, // s_mov_b32 s12, 0
            // loop 0: s9 ^= (v0 == s12)
            0x7D84_000C, // v_cmp_eq_u32 vcc_lo, s12, v0
            0x8909_096A, // s_xor_b32 s9, vcc_lo, s9
            0x800C_810C, // s_add_u32 s12, s12, 1
            0xBF0A_820C, // s_cmp_lt_u32 s12, 2
            0xBF85_FFFB, // s_cbranch_scc1 -> loop 0
            // data phi s7: joins constant 1 with loop 0 mask s9
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0002, // s_cbranch_scc1 -> else0
            0xBE87_0381, // s_mov_b32 s7, 1
            0xBF82_0001, // s_branch -> loop1_init
            0xBE87_0309, // else0: s_mov_b32 s7, s9
            // loop 1: s2 ^= (v0 == s7) -> reading s2 as data discovers s7 then s9
            0xBE82_0380, // loop1_init: s_mov_b32 s2, 0
            0xBE8C_0380, // s_mov_b32 s12, 0
            0x7D84_0007, // loop 1: v_cmp_eq_u32 vcc_lo, s7, v0
            0x8902_026A, // s_xor_b32 s2, vcc_lo, s2
            0x800C_810C, // s_add_u32 s12, s12, 1
            0xBF0A_820C, // s_cmp_lt_u32 s12, 2
            0xBF85_FFFB, // s_cbranch_scc1 -> loop 1
            // data phi s6: joins 0x0BAD_F00D with loop 1 mask s2
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF85_0003, // s_cbranch_scc1 -> else1
            0xBE86_03FF,
            0x0BAD_F00D, // s_mov_b32 s6, 0x0BAD_F00D
            0xBF82_0001, // s_branch -> end
            0xBE86_0302, // else1: s_mov_b32 s6, s2
            0x7E02_0206, // end: v_mov_b32 v1, s6
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 2).expect("lower");
        assert!(lowered.wave_ops);
        assert_eq!(lowered.wave_ballots, 2);
        validate_dxil(&lowered.dxil);
    }
}
