//! Binary64 sums whole-wave sections leave ([`crate::emit::wave::Shape::Sum64`]):
//! the console compiler's expansion of a wave sum of doubles, lowered to one
//! DXIL wave operation on `double`.

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V};
use crate::emit::shader::flags;
use crate::ir::Value;

impl Lower<'_> {
    /// Dword `high` (the upper, or the lower) of the binary64 sum over the
    /// host's active lanes of `mask ? (lo, hi) : +0.0` ([`crate::emit::wave::Shape::Sum64`]).
    // shortcut: the host adds in its own order, the console in its tree of
    // DPP steps; terms that are not exact in binary64 may round differently.
    pub(super) fn sum64(
        &mut self,
        lo: Value,
        hi: Value,
        mask: Value,
        high: bool,
    ) -> Result<V, Why> {
        let c = self.cur;
        if let Some(&r) = self.sums64.get(&(c, lo, hi, mask)) {
            let (lo, hi) = self.split64(r);
            return Ok(if high { hi } else { lo });
        }
        let key = (c, lo, hi, mask);
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let (lo, hi) = (self.i32_of(lo)?, self.i32_of(hi)?);
        let i64t = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        let bits = self.s.m.bin(c, Bin::Or, lo, hi);
        let zero = self.s.m.int_const(i64t, 0);
        let bits = self.s.m.select(c, on, bits, zero);
        self.s.flags |= flags::WAVE_OPS | flags::FLOAT64 | flags::INT64;
        let double = self.s.m.double();
        let x = self.s.m.cast(c, Cast::Bitcast, bits, double);
        let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
        let r = self.s.op(
            c,
            "waveActiveOp.f64",
            Attr::NoUnwind,
            double,
            119,
            &[x, sum, signed],
        );
        let r = self.s.m.cast(c, Cast::Bitcast, r, i64t);
        self.sums64.insert(key, r);
        let (lo, hi) = self.split64(r);
        Ok(if high { hi } else { lo })
    }
}
