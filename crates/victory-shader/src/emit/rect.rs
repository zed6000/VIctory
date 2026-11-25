//! The console's rectangle lists on the host. A rectangle comes as three
//! corners in any order (the blits put the right angle at `v1`, the tiled
//! lighting at `v0`), and the hardware adds the fourth opposite the right
//! angle `vc`, at `va + vb - vc`, for the position and every attribute. The
//! host has no such primitive; a geometry shader made for the vertex shader
//! that feeds it takes each triangle `(v0, v1, v2)`, finds the right angle
//! opposite its longest side, and emits the triangle with its completion
//! `(va, vb, v3)` (`va` before `vc`, `vb` after it), both turning the same
//! way. Integer outputs (which can only be flat) keep `va`'s value at the
//! fourth corner.

use super::bitcode::{Attr, Bin, Term, V, pred};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// Which of a rectangle's three corners (x and y of each) holds the right
/// angle: the one opposite the longest side, else (ties, NaN) `1`. The
/// geometry shader of [`rectangle_gs`] makes the same choice on the GPU.
pub fn right_angle(xy: [[f32; 2]; 3]) -> usize {
    let side = |i: usize, j: usize| {
        let (dx, dy) = (xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]);
        dx * dx + dy * dy
    };
    let (d12, d20, d01) = (side(1, 2), side(2, 0), side(0, 1));
    if d12 > d20 && d12 > d01 {
        0
    } else if d01 > d12 && d01 > d20 {
        2
    } else {
        1
    }
}

/// The geometry shader that completes the rectangles of the vertex shader
/// container `vs` (its output signature is the shader's input and output).
pub fn rectangle_gs(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let part = c
        .part(FourCC::OSG1)
        .ok_or(ShaderError::Malformed("a vertex shader without outputs"))?;
    let mut elements = super::signature(part.data)?;
    elements.retain(|e| e.row.is_some());
    for e in &mut elements {
        e.usage = (1 << e.cols) - 1;
    }
    let mut s = Shader::new(false);
    s.gs = Some(6);
    let b = s.entry;
    // Corner k's values, element by element, component by component.
    let mut corners: Vec<Vec<Vec<super::bitcode::V>>> = Vec::new();
    for k in 0..3u32 {
        let mut values = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let (ty, suffix) = if e.comp == 9 {
                (s.t.f32, ".f32")
            } else {
                (s.t.i32, ".i32")
            };
            let mut comps = Vec::new();
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
                comps.push(s.op(b, &format!("loadInput{suffix}"), Attr::ReadNone, ty, 4, &args));
            }
            values.push(comps);
        }
        corners.push(values);
    }
    // The right angle, as [`right_angle`] picks it from the position.
    let position = elements
        .iter()
        .position(|e| e.kind == 3 && e.cols >= 2 && e.comp == 9)
        .ok_or(ShaderError::Malformed("a rectangle list without a position"))?;
    let side = |s: &mut Shader, i: usize, j: usize| {
        let mut sum = None;
        for (&p, &q) in corners[i][position].iter().zip(&corners[j][position]).take(2) {
            let d = s.m.bin(b, Bin::Sub, p, q);
            let sq = s.m.bin(b, Bin::Mul, d, d);
            sum = Some(match sum {
                Some(x) => s.m.bin(b, Bin::Add, x, sq),
                None => sq,
            });
        }
        sum.expect("two columns")
    };
    let (d12, d20, d01) = (side(&mut s, 1, 2), side(&mut s, 2, 0), side(&mut s, 0, 1));
    let longest = |s: &mut Shader, d: V, others: [V; 2]| {
        let [x, y] = others.map(|o| s.m.cmp(b, pred::OGT, d, o));
        s.m.bin(b, Bin::And, x, y)
    };
    let (at0, at2) = (longest(&mut s, d12, [d20, d01]), longest(&mut s, d01, [d12, d20]));
    // `x0` if the right angle is at v0, `x2` if at v2, else `x1`.
    let pick = |s: &mut Shader, [x0, x1, x2]: [V; 3]| {
        let x = s.m.select(b, at2, x2, x1);
        s.m.select(b, at0, x0, x)
    };
    // The completing triangle (va, vb, v3), element by element.
    let mut second: [Vec<Vec<V>>; 3] = Default::default();
    for (id, e) in elements.iter().enumerate() {
        let mut cols: [Vec<V>; 3] = Default::default();
        let [c0, c1, c2] = [0, 1, 2].map(|k| &corners[k][id]);
        for ((&v0, &v1), &v2) in c0.iter().zip(c1).zip(c2) {
            let va = pick(&mut s, [v2, v0, v1]);
            let vb = pick(&mut s, [v1, v2, v0]);
            let v3 = if e.comp == 9 {
                let vc = pick(&mut s, [v0, v1, v2]);
                let sum = s.m.bin(b, Bin::Add, va, vb);
                s.m.bin(b, Bin::Sub, sum, vc)
            } else {
                va
            };
            for (k, v) in [va, vb, v3].into_iter().enumerate() {
                cols[k].push(v);
            }
        }
        for (k, c) in cols.into_iter().enumerate() {
            second[k].push(c);
        }
    }
    corners.extend(second);
    for triangle in [[0, 1, 2], [3, 4, 5]] {
        for k in triangle {
            for (id, e) in elements.iter().enumerate() {
                let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
                for col in 0..e.cols {
                    let v = corners[k][id][usize::from(col)];
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                    s.op(b, &format!("storeOutput{suffix}"), Attr::NoUnwind, s.t.void, 5, &args);
                }
            }
            let stream = [s.m.i8(0)];
            s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
        }
        let stream = [s.m.i8(0)];
        s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    }
    s.inputs = elements.clone();
    s.outputs = elements;
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(test)]
mod rule_tests {
    use super::right_angle;

    /// The right angle is found wherever it is; triangles that are no
    /// rectangle's take the corner opposite their longest side (their
    /// widest angle), and ties and NaN keep the blits' corner, `v1`.
    #[test]
    fn right_angles_are_found_in_any_order() {
        let (tl, bl, br, tr) = ([0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]);
        assert_eq!(right_angle([tl, bl, br]), 1);
        assert_eq!(right_angle([tl, bl, tr]), 0);
        assert_eq!(right_angle([bl, tr, tl]), 2);
        assert_eq!(right_angle([[0.0, 0.0], [3.0, 1.0], [1.0, 0.0]]), 2, "obtuse at v2");
        assert_eq!(right_angle([tl, tl, tl]), 1, "a point");
        assert_eq!(right_angle([tl, [f32::NAN, 0.0], br]), 1);
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;

    fn element(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A shader copying its inputs to its outputs (vertex), or writing
    /// target 0 from TEXCOORD (pixel).
    fn shader(pixel: bool) -> Vec<u8> {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let (inputs, outputs) = if pixel {
            (
                vec![element("SV_Position", 3, 4, 0, 4), element("TEXCOORD", 0, 2, 1, 2)],
                vec![element("SV_Target", 16, 0, 0, 4)],
            )
        } else {
            (
                vec![element("POSITION", 0, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
                vec![element("SV_Position", 3, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
            )
        };
        let load = |s: &mut Shader, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let values: Vec<_> = if pixel {
            let one = s.m.f32(1f32.to_bits());
            vec![load(&mut s, 1, 0), load(&mut s, 1, 1), one, one]
        } else {
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
                .into_iter()
                .map(|(id, c)| load(&mut s, id, c))
                .collect()
        };
        let mut k = 0;
        for (id, e) in outputs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), values[k]];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
                k += 1;
            }
        }
        s.inputs = inputs;
        s.outputs = outputs;
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// One rectangle with a coordinate attribute, its right angle at each of
    /// the three corners in turn (the blits' top left, bottom left, bottom
    /// right; the tiled lighting's top left, bottom left, top right; and
    /// one with it last): every pixel of the square is drawn, the attribute
    /// interpolated over it, the missing half included. Three corners on one
    /// line draw nothing.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let (vs, ps) = (shader(false), shader(true));
        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) =
            super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref())
        {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        let corner = |x: f32, y: f32| [x, y, 0.5, 1.0, x, y];
        let (tl, bl, br, tr) = (corner(-0.5, 0.5), corner(-0.5, -0.5), corner(0.5, -0.5), corner(0.5, 0.5));
        let orders = [[tl, bl, br], [tl, bl, tr], [bl, tr, tl], [tl, corner(0.0, 0.0), br]];
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            for (n, order) in orders.iter().enumerate() {
                let vertices = order.concat();
                let draw = Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 24,
                    size: (32, 32),
                    mesh_groups: 0,
                };
                let texels = host.draw(&root, &pso, &draw).unwrap();
                for (i, t) in texels.iter().enumerate() {
                    let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                    let (u, v) = ((x + 0.5) / 16.0 - 1.0, 1.0 - (y + 0.5) / 16.0);
                    let inside = n < 3 && u.abs() < 0.5 && v.abs() < 0.5;
                    let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                    let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                    assert!(near, "warp {warp} order {n} pixel ({x}, {y}): {t:?}, want {want:?}");
                }
            }
        }
    }
}
