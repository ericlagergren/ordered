#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{
    fmt, mem,
    ops::{BitAnd, BitOr, BitXor, Not},
};

mod private {
    pub trait Ord {
        const ORD: u8;
    }

    pub trait Encode {
        fn append<E, O>(&self, e: &mut E)
        where
            E: Extend<u8>,
            O: Ord;
    }
}
use private::{Encode, Ord};

/// Implemented by types TODO
pub trait Ordered: Sized + Encode + Ord {
    /// Encodes itself into `e`.
    fn encode<E: Extend<u8>>(&self, e: &mut E) {
        Encode::append::<_, Self>(self, e)
    }

    /// Encodes itself and returns the result as a `Vec<u8>`.
    #[cfg(feature = "alloc")]
    fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// Encodes itself in reverse order.
    fn rev(self) -> Rev<Self> {
        Rev { v: self }
    }
}

impl<T> Ordered for T where T: Sized + Encode + Ord {}

/// Encodes `T` in reverse order.
///
/// See [`Ordered::rev`].
#[repr(transparent)]
pub struct Rev<T> {
    v: T,
}

impl<T> Rev<T> {
    /// Returns the underlying value.
    pub fn into_inner(self) -> T {
        self.v
    }
}

impl<T: Ord> Ord for Rev<T> {
    const ORD: u8 = !T::ORD;
}

impl<T: Encode> Encode for Rev<T> {
    fn append<E, O>(&self, e: &mut E)
    where
        E: Extend<u8>,
        O: Ord,
    {
        self.v.append::<_, O>(e)
    }
}

macro_rules! ord {
    ($($name:ty),+ $(,)?) => {
        $(
            impl Ord for $name {
                const ORD: u8 = 0x00;
            }
        )+
    };
}
ord! {
    (),
    f32, f64,
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    char,
    bool,
    &str,
}

impl<T: Ord> Ord for &[T] {
    const ORD: u8 = 0x00;
}

impl<T: Ord, const N: usize> Ord for &[T; N] {
    const ORD: u8 = 0x00;
}

impl Encode for &str {
    fn append<E, O>(&self, e: &mut E)
    where
        E: Extend<u8>,
        O: Ord,
    {
        e.extend([Op::STRING.as_u8()]);
        for chunk in self.split_inclusive('\x00') {
            e.extend(chunk.as_bytes().iter().map(|v| v ^ O::ORD));
            if chunk.ends_with('\x00') {
                e.extend([0xff ^ O::ORD]);
            }
        }
        e.extend([O::ORD, O::ORD]);
    }
}

impl Encode for &[u8] {
    fn append<E, O>(&self, e: &mut E)
    where
        E: Extend<u8>,
        O: Ord,
    {
        e.extend([Op::STRING.as_u8()]);
        for chunk in self.split_inclusive(|c| *c == 0) {
            e.extend(chunk.iter().map(|v| v ^ O::ORD));
            if chunk.ends_with(&[0]) {
                e.extend([0xff ^ O::ORD]);
            }
        }
        e.extend([O::ORD, O::ORD]);
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Op(u8);

impl Op {
    const STRING: Self = Self(0x01);
    const FLOAT32: Self = Self(0x02);
    const FLOAT64: Self = Self(0x03);
    const INT: Self = Self(0x20);
    const POS_INT: Self = Self(0x10);
    const INF: Self = Self(0xff);

    const fn as_u8(self) -> u8 {
        self.0
    }
}

impl PartialEq<u8> for Op {
    fn eq(&self, other: &u8) -> bool {
        self.0 == *other
    }
}

impl BitAnd for Op {
    type Output = Self;
    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0 & other.0)
    }
}
impl BitAnd<u8> for Op {
    type Output = Self;
    fn bitand(self, other: u8) -> Self::Output {
        Self(self.0 & other)
    }
}

impl BitOr for Op {
    type Output = Self;
    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0 | other.0)
    }
}
impl BitOr<u8> for Op {
    type Output = Self;
    fn bitor(self, other: u8) -> Self::Output {
        Self(self.0 | other)
    }
}

impl BitXor for Op {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0 ^ other.0)
    }
}
impl BitXor<u8> for Op {
    type Output = Self;
    fn bitxor(self, other: u8) -> Self::Output {
        Self(self.0 ^ other)
    }
}

impl Not for Op {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut op = *self;
        if op & 0x80 != 0 {
            write!(f, "rev ")?;
            op = !op;
        }
        if op & 0xe0 == Self::INT {
            op = Self::INT;
        }
        match op {
            Self::STRING => write!(f, "string"),
            Self::FLOAT32 => write!(f, "float32"),
            Self::FLOAT64 => write!(f, "float64"),
            Self::INT => write!(f, "int"),
            Self::INF => write!(f, "infinity"),
            op => write!(f, "opcode({:x})", op.0),
        }
    }
}

#[non_exhaustive]
pub struct Infinity;

pub enum StringOrInf {
    String,
    Inf,
}

// impl Sealed for Infinity {
//     fn encode<W: Write>(&self, w: &mut W) -> Result<(), W::Error> {
//         w.write_all(&[Op::INF.as_u8()])?;
//         Ok(())
//     }
// }
// impl Encode for Infinity {}

macro_rules! encode_float{
    ($($name:ident),+) => {
        $(
            impl Encode for $name {
                fn append<E, O>(&self, e: &mut E)
                where
                    E: Extend<u8>,
                    O: Ord,
                {
                    const BITS: usize = mem::size_of::<$name>();

                    let mut u = self.to_bits();
                    if self.is_nan() {
                        // NaN maps beyond -inf.
                        u = !u;
                    }
                    u &= !(1 << (BITS-1))
                        | (((u as i64) as u64) >> (BITS-1));

                    let mut buf = u.to_be_bytes();
                    for c in &mut buf {
                        *c ^= O::ORD;
                    }

                    e.extend([Op::FLOAT64.as_u8() ^ O::ORD]);
                    e.extend(buf);
                }
            }
        )+
    };
}
encode_float!(f32, f64);

macro_rules! encode_uint {
    ($($name:ident),+) => {
        $(
            impl Encode for $name {
                fn append<E, O>(&self, e: &mut E)
                where
                    E: Extend<u8>,
                    O: Ord,
                {
                    let n = if *self == 0 {
                        1
                    } else {
                        (((self.ilog2() + 1) + 7) / 8).max(1) as usize
                    };

                    let mut buf = [0u8; (($name::BITS/8)+1) as usize];
                    for (i, c) in buf.iter_mut().rev().take(n).enumerate() {
                        let s = (i*8) as u32;
                        *c = (self.wrapping_shr(s) as u8) ^ O::ORD;
                    }
                    buf[8-n] = (((n-1) as u8) |
                        (Op::INT|Op::POS_INT).as_u8()) ^ O::ORD;

                    if let Some(buf) = buf.get(8-n..) {
                        e.extend(buf.iter().copied())
                    }
                }
            }
        )+
    };
}
encode_uint!(u8, u16, u32, u64, u128, usize);

macro_rules! encode_int {
    ($($name:ident),+) => {
        $(
            impl Encode for $name {
                fn append<E, O>(&self, e: &mut E)
                where
                    E: Extend<u8>,
                    O: Ord,
                {
                    if *self >= 0 {
                        return self.unsigned_abs().encode(e);
                    }
                    let u = (!*self).unsigned_abs();

                    // Max number of bytes used by this integer.
                    let n = if u == 0 {
                        1
                    } else {
                        (((u.ilog2() + 1) + 7) / 8).max(1) as usize
                    };

                    let mut buf = [0u8; (($name::BITS/8)+1) as usize];
                    for (i, c) in buf.iter_mut().rev().take(n).enumerate() {
                        let s = (i*8) as u32;
                        *c = (u.wrapping_shr(s) as u8) ^ 0xff ^ O::ORD;
                    }
                    buf[8-n] = (((n-1) as u8) | Op::INT.as_u8()) ^ 0xf ^ O::ORD;

                    if let Some(buf) = buf.get(8-n..) {
                        e.extend(buf.iter().copied())
                    }
                }
            }
        )+
    };
}
encode_int!(i8, i16, i32, i64, i128, isize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_encoding() {
        let cases: &[(&str, &[u8])] = &[
            ("", b"\x01\x00\x00"),
            ("hello\x00world", b"\x01hello\x00\xffworld\x00\x00"),
            ("hello", b"\x01hello\x00\x00"),
        ];
        for (i, (input, want)) in cases.into_iter().enumerate() {
            let got = input.encode_to_vec();
            assert_eq!(got, *want, "#{i}");

            let got = input.as_bytes().encode_to_vec();
            assert_eq!(got, *want, "#{i} (bytes)");
        }
    }

    #[test]
    fn test_u64_encoding() {
        let cases: &[(u64, &[u8])] = &[
            (0x12345, b"\x32\x01\x23\x45"),
            (0x12345678, b"\x33\x12\x34\x56\x78"),
            (0x123456789abcdef0, b"\x37\x12\x34\x56\x78\x9a\xbc\xde\xf0"),
            (0x2345, b"\x31\x23\x45"),
            (0x45, b"\x30\x45"),
            (0xffffffffffffffff, b"\x37\xff\xff\xff\xff\xff\xff\xff\xff"),
        ];
        for (i, (input, want)) in cases.into_iter().enumerate() {
            let got = input.encode_to_vec();
            assert_eq!(got, *want, "#{i}");
        }
    }

    #[test]
    fn test_i64_encoding() {
        let cases: &[(i64, &[u8])] = &[
            (0x7fffffffffffffff, b"\x37\x7f\xff\xff\xff\xff\xff\xff\xff"),
            (-0x7fffffffffffffff, b"\x28\x80\x00\x00\x00\x00\x00\x00\x01"),
            (-0x8000000000000000, b"\x28\x80\x00\x00\x00\x00\x00\x00\x00"),
            (0x123456789abcdef0, b"\x37\x12\x34\x56\x78\x9a\xbc\xde\xf0"),
            (-0x123456789abcdef0, b"\x28\xed\xcb\xa9\x87\x65\x43\x21\x10"),
            (0x12345678, b"\x33\x12\x34\x56\x78"),
            (-0x12345678, b"\x2c\xed\xcb\xa9\x88"),
            (0x2345, b"\x31\x23\x45"),
            (-0x2345, b"\x2e\xdc\xbb"),
            (0x45, b"\x30\x45"),
            (-0x45, b"\x2f\xbb"),
            (0x12345, b"\x32\x01\x23\x45"),
            (-0x12345, b"\x2d\xfe\xdc\xbb"),
        ];
        for (i, (input, want)) in cases.into_iter().enumerate() {
            let got = input.encode_to_vec();
            assert_eq!(got, *want, "#{i}");
        }
    }

    #[test]
    fn test_rev() {
        let cases: &[(&str, &[u8])] = &[
            ("", b"\x01\x00\x00"),
            ("hello\x00world", b"\x01hello\x00\xffworld\x00\x00"),
            ("hello", b"\x01hello\x00\x00"),
        ];
        for (i, (input, want)) in cases.into_iter().enumerate() {
            let got = input.rev().rev().encode_to_vec();
            assert_eq!(got, *want, "#{i}");

            let got = input.as_bytes().rev().rev().encode_to_vec();
            assert_eq!(got, *want, "#{i} (bytes)");
        }
    }
}
