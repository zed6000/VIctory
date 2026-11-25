//! The console's rectangle lists on the host. A rectangle comes as three
//! corners, the second between the other two, and the hardware adds the
//! fourth at `v0 + v2 - v1`, for the position and every attribute. The host
//! has no such primitive; a geometry shader made for the vertex shader that
//! feeds it takes each triangle `(v0, v1, v2)` and emits it with its
//! completion `(v0, v2, v3)`, both turning the same way. Integer outputs
//! (which can only be flat) keep `v0`'s value at the fourth corner.

use super::bitcode::{Attr, Bin, Term};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

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
    let fourth: Vec<Vec<_>> = elements
        .iter()
        .enumerate()
        .map(|(id, e)| {
            (0..usize::from(e.cols))
                .map(|col| {
                    let [v0, v1, v2] = [0, 1, 2].map(|k| corners[k][id][col]);
                    if e.comp == 9 {
                        let sum = s.m.bin(b, Bin::Add, v0, v2);
                        s.m.bin(b, Bin::Sub, sum, v1)
                    } else {
                        v0
                    }
                })
                .collect()
        })
        .collect();
    corners.push(fourth);
    for triangle in [[0, 1, 2], [0, 2, 3]] {
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

    /// One rectangle (top left, bottom left, bottom right) with a
    /// coordinate attribute: every pixel of the square is drawn, the
    /// attribute interpolated over it, the top-right half included.
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
        #[rustfmt::skip]
        let vertices = [
            -0.5, 0.5, 0.5, 1.0, -0.5, 0.5,
            -0.5, -0.5, 0.5, 1.0, -0.5, -0.5,
            0.5, -0.5, 0.5, 1.0, 0.5, -0.5,
        ];
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
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
                let inside = u.abs() < 0.5 && v.abs() < 0.5;
                let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(near, "warp {warp} pixel ({x}, {y}): {t:?}, want {want:?}");
            }
        }
    }
}
