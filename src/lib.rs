#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::NonZero,
    ops::{Index, Range},
};

pub trait Id: Copy + Ord {
    const MAX: usize;

    /// Convert to this type from a `usize`.
    ///
    /// `idx` is guaranteed to be less than `Self::MAX`.
    fn from_usize(idx: usize) -> Self;

    /// Convert to `usize` from this type.
    ///
    /// The returned value should be less than `Self::MAX`.
    fn into_usize(self) -> usize;
}

macro_rules! impl_idx_for_nums {
    ($($ty:ty),*) => {$(
        impl Id for $ty {
            const MAX: usize = <$ty>::MAX as usize;
            #[inline]
            fn from_usize(idx: usize) -> Self {
                idx as $ty
            }
            #[inline]
            fn into_usize(self) -> usize {
                self as usize
            }
        }
        impl Id for NonZero<$ty> {
            const MAX: usize = (<$ty>::MAX - 1) as usize;
            #[inline]
            fn from_usize(idx: usize) -> Self {
                unsafe { NonZero::new_unchecked((idx + 1) as $ty) }
            }
            #[inline]
            fn into_usize(self) -> usize {
                self.get() as usize
            }
        }
    )*};
}
impl_idx_for_nums!(u8, u16, u32, u64, usize);

pub struct Idx<T, I: Id> {
    raw: I,
    phantom: PhantomData<fn() -> T>,
}

impl<T, I: Id> Idx<T, I> {
    #[inline]
    pub const fn into_raw(self) -> I {
        self.raw
    }
}

impl<T, I: Id> Clone for Idx<T, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, I: Id> Copy for Idx<T, I> {}

impl<T, I: Id + fmt::Debug> fmt::Debug for Idx<T, I> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut type_name = core::any::type_name::<T>();
        if let Some(idx) = type_name.rfind(':') {
            type_name = &type_name[idx + 1..]
        }
        write!(fmt, "Idx::<{}>({:?})", type_name, self.raw)
    }
}

impl<T, I: Id> PartialEq for Idx<T, I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<T, I: Id> Eq for Idx<T, I> {}

impl<T, I: Id> Ord for Idx<T, I> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<T, I: Id> PartialOrd for Idx<T, I> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, I: Id + Hash> Hash for Idx<T, I> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state)
    }
}

pub struct IdxRange<T, I: Id> {
    start: I,
    end: I,
    phantom: PhantomData<fn() -> T>,
}

impl<T, I: Id> IdxRange<T, I> {
    #[inline]
    pub const fn new(range: Range<I>) -> Self {
        Self { start: range.start, end: range.end, phantom: PhantomData }
    }

    #[inline]
    pub const fn start(&self) -> I {
        self.start
    }

    #[inline]
    pub const fn end(&self) -> I {
        self.end
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.end.into_usize() - self.start.into_usize()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

impl<T, I: Id + fmt::Debug> fmt::Debug for IdxRange<T, I> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut type_name = core::any::type_name::<T>();
        if let Some(idx) = type_name.rfind(':') {
            type_name = &type_name[idx + 1..]
        }
        write!(fmt, "IdxRange::<{}>({:?}..{:?})", type_name, self.start, self.end)
    }
}

impl<T, I: Id> Clone for IdxRange<T, I> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, I: Id> Copy for IdxRange<T, I> {}

impl<T, I: Id> PartialEq for IdxRange<T, I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}
impl<T, I: Id> Eq for IdxRange<T, I> {}

impl<T, I: Id> Ord for IdxRange<T, I> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.start.cmp(&other.start).then_with(|| self.end.cmp(&other.end))
    }
}

impl<T, I: Id> PartialOrd for IdxRange<T, I> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Arena<T, I: Id> {
    data: Vec<T>,
    phantom: PhantomData<(I, T)>,
}

impl<T, I: Id> Arena<T, I> {
    #[inline]
    pub const fn new() -> Self {
        Self { data: Vec::new(), phantom: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity), phantom: PhantomData }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn alloc(&mut self, value: T) -> Idx<T, I> {
        self.try_alloc(value).expect("arena is full")
    }

    #[inline]
    pub fn try_alloc(&mut self, value: T) -> Option<Idx<T, I>> {
        if self.data.len() > I::MAX {
            None
        } else {
            let idx = I::from_usize(self.data.len());
            self.data.push(value);
            Some(Idx { raw: idx, phantom: PhantomData })
        }
    }

    #[inline]
    pub fn alloc_many(&mut self, values: impl IntoIterator<Item = T>) -> IdxRange<T, I> {
        self.try_alloc_many(values).expect("arena is full")
    }

    #[inline]
    pub fn try_alloc_many(
        &mut self,
        values: impl IntoIterator<Item = T>,
    ) -> Option<IdxRange<T, I>> {
        let start = I::from_usize(self.data.len());
        let mut len = 0;
        for value in values {
            if len > I::MAX {
                return None;
            }
            self.data.push(value);
            len += 1;
        }
        let end = I::from_usize(len);
        Some(IdxRange::new(start..end))
    }
}

impl<T, I: Id> Default for Arena<T, I> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, I: Id> Index<Idx<T, I>> for Arena<T, I> {
    type Output = T;

    #[inline]
    fn index(&self, idx: Idx<T, I>) -> &Self::Output {
        &self.data[idx.raw.into_usize()]
    }
}

impl<T, I: Id> Index<IdxRange<T, I>> for Arena<T, I> {
    type Output = [T];

    #[inline]
    fn index(&self, range: IdxRange<T, I>) -> &Self::Output {
        &self.data[range.start.into_usize()..range.end.into_usize()]
    }
}

impl<T: Clone, I: Id> Clone for Arena<T, I> {
    #[inline]
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), phantom: PhantomData }
    }
}

impl<T: PartialEq, I: Id> PartialEq for Arena<T, I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq, I: Id> Eq for Arena<T, I> {}

impl<T: Hash, I: Id> Hash for Arena<T, I> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}
