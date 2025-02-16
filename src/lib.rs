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
    ops::{Index, IndexMut, Range},
};

/// A trait for index types used in arenas.
///
/// An [`Id`] represents both the internal index in an arena and a type-level distinction
/// (for example, when using multiple arenas with the same underlying numeric index type).
pub trait Id: Copy + Ord {
    /// The maximum value (as a usize) this id type can represent.
    const MAX: usize;

    /// Converts a `usize` value to this id type.
    ///
    /// The input `idx` (should / is guaranteed to) be less than `Self::MAX`.
    fn from_usize(idx: usize) -> Self;

    /// Converts this id type into a `usize`.
    ///
    /// The returned value (should / is guaranteed to) be less than `Self::MAX`.
    fn into_usize(self) -> usize;
}

macro_rules! impl_id_for_nums {
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
                (self.get() - 1) as usize
            }
        }
    )*};
}
impl_id_for_nums!(u8, u16, u32, u64, usize);

/// A typed index for referencing elements in an [`Arena`].
///
/// The [`Idx<T, I>`] type wraps an underlying id of type `I` and carries a phantom type `T`
/// to ensure type safety when indexing into an arena.
///
/// # Examples
///
/// ```
/// use indexed_arena::{Arena, Idx};
///
/// let mut arena: Arena<&str, u32> = Arena::new();
/// let idx: Idx<&str, u32> = arena.alloc("hello");
/// assert_eq!(arena[idx], "hello");
/// ```
pub struct Idx<T, I: Id> {
    raw: I,
    phantom: PhantomData<fn() -> T>,
}

impl<T, I: Id> Idx<T, I> {
    /// Consumes the index and returns its underlying raw value.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexed_arena::{Arena, Idx};
    ///
    /// let mut arena: Arena<i32, u32> = Arena::new();
    /// let idx = arena.alloc(10);
    /// let raw = idx.into_raw();
    /// // raw is a u32 representing the index inside the arena.
    /// assert_eq!(raw, 0u32);
    /// ```
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

/// A range of indices within an `Arena`.
///
/// This type represents a contiguous range of allocated indices in an arena.
///
/// # Examples
///
/// ```
/// use indexed_arena::{Arena, IdxRange};
///
/// let mut arena: Arena<i32, u32> = Arena::new();
/// let range: IdxRange<i32, u32> = arena.alloc_many(vec![1, 2, 3, 4]);
/// assert_eq!(range.len(), 4);
/// assert!(!range.is_empty());
/// ```
pub struct IdxRange<T, I: Id> {
    start: I,
    end: I,
    phantom: PhantomData<fn() -> T>,
}

impl<T, I: Id> IdxRange<T, I> {
    /// Creates a new [`IdxRange`] from the given range of raw indices.
    #[inline]
    pub const fn new(range: Range<I>) -> Self {
        Self { start: range.start, end: range.end, phantom: PhantomData }
    }

    /// Returns the starting raw index.
    #[inline]
    pub const fn start(&self) -> I {
        self.start
    }

    /// Returns the ending raw index.
    #[inline]
    pub const fn end(&self) -> I {
        self.end
    }

    /// Returns the number of indices in the range.
    #[inline]
    pub fn len(&self) -> usize {
        self.end.into_usize() - self.start.into_usize()
    }

    /// Returns true if the range is empty.
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

/// A index-based arena.
///
/// [`Arena`] provides a mechanism to allocate objects and refer to them by a
/// strongly-typed index ([`Idx<T, I>`]). The index not only represents the position
/// in the underlying vector but also leverages the type system to prevent accidental misuse
/// across different arenas.
pub struct Arena<T, I: Id> {
    data: Vec<T>,
    phantom: PhantomData<(I, T)>,
}

impl<T, I: Id> Arena<T, I> {
    /// Creates a new empty arena.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexed_arena::Arena;
    ///
    /// let arena: Arena<i32, u32> = Arena::new();
    /// assert!(arena.is_empty());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self { data: Vec::new(), phantom: PhantomData }
    }

    /// Creates a new arena with the specified capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexed_arena::Arena;
    ///
    /// let arena: Arena<i32, u32> = Arena::with_capacity(10);
    /// assert!(arena.is_empty());
    /// assert!(arena.capacity() >= 10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity), phantom: PhantomData }
    }

    /// Returns the number of elements stored in the arena.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the capacity of the arena.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns `true` if the arena contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Allocates an element in the arena and returns its index.
    ///
    /// # Panics
    ///
    /// Panics if the arena is full (i.e. if the number of elements exceeds `I::MAX`).
    /// If you hnadle this case, use [`Arena::try_alloc`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexed_arena::{Arena, Idx};
    ///
    /// let mut arena: Arena<&str, u32> = Arena::new();
    /// let idx: Idx<&str, u32> = arena.alloc("hello");
    /// assert_eq!(arena[idx], "hello");
    /// ```
    #[inline]
    pub fn alloc(&mut self, value: T) -> Idx<T, I> {
        self.try_alloc(value).expect("arena is full")
    }

    /// Fallible version of [`Arena::alloc`].
    ///
    /// This method returns `None` if the arena is full.
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

    /// Allocates multiple elements in the arena and returns the index range covering them.
    ///
    /// # Panics
    ///
    /// Panics if the arena cannot allocate all elements (i.e. if the arena becomes full).
    ///
    /// # Examples
    ///
    /// ```
    /// use indexed_arena::{Arena, IdxRange};
    ///
    /// let mut arena: Arena<i32, u32> = Arena::new();
    /// let range: IdxRange<i32, u32> = arena.alloc_many(vec![10, 20, 30]);
    /// assert_eq!(&arena[range], &[10, 20, 30]);
    /// ```
    #[inline]
    pub fn alloc_many(&mut self, values: impl IntoIterator<Item = T>) -> IdxRange<T, I> {
        self.try_alloc_many(values).expect("arena is full")
    }

    /// Fallible version of [`Arena::alloc_many`].
    ///
    /// This method returns `None` if the arena becomes full.
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

    /// Shrinks the capacity of the arena to fit the number of elements.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
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

impl<T, I: Id> IndexMut<Idx<T, I>> for Arena<T, I> {
    #[inline]
    fn index_mut(&mut self, index: Idx<T, I>) -> &mut Self::Output {
        &mut self.data[index.raw.into_usize()]
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
