use indexed_arena::{Arena, Id, IdxSpan};
use std::{fmt::Debug, num::NonZero};

macro_rules! mktest {
    ($name:ident, $expr:expr) => {
        #[test]
        fn $name() {
            $expr
        }
    };
}
// The `Debug` output embeds `core::any::type_name`, whose exact form is
// unspecified, so only the stable parts are checked: the `Idx::<` prefix and the
// raw index. A plain integer id stores index 0 as `0`, while a `NonZero` id stores
// it as `1` (its niche offsets by one), which exercises the round-trip contract.
fn construct<T: Id + Debug>(raw_repr: &str) {
    let mut arena = Arena::<String, T>::new();
    assert_eq!(arena.len(), 0);
    assert!(arena.is_empty());
    let idx = arena.alloc("foo".to_string());
    assert_eq!(arena.len(), 1);
    assert!(!arena.is_empty());
    let debug = format!("{idx:?}");
    assert!(debug.starts_with("Idx::<"), "{debug}");
    assert!(debug.ends_with(&format!("({raw_repr})")), "{debug}");
}
mktest!(construct_u8, construct::<u8>("0"));
mktest!(construct_u16, construct::<u16>("0"));
mktest!(construct_u32, construct::<u32>("0"));
mktest!(construct_u64, construct::<u64>("0"));
mktest!(construct_usize, construct::<usize>("0"));
mktest!(construct_nz_u8, construct::<NonZero<u8>>("1"));
mktest!(construct_nz_u16, construct::<NonZero<u16>>("1"));
mktest!(construct_nz_u32, construct::<NonZero<u32>>("1"));
mktest!(construct_nz_u64, construct::<NonZero<u64>>("1"));
mktest!(construct_nz_usize, construct::<NonZero<usize>>("1"));

#[test]
fn alloc_get_iter() {
    #[derive(PartialEq)]
    struct T(u32);
    let mut arena = Arena::<_, u32>::new();
    let idx1 = arena.alloc(T(42));
    assert_eq!(idx1.into_raw(), 0);
    let idx2 = arena.alloc(T(17));
    assert_eq!(idx2.into_raw(), 1);
    let idx3 = arena.alloc(T(17));
    assert_eq!(idx3.into_raw(), 2);
    assert_eq!(arena[idx1].0, 42);
    assert_eq!(arena[idx2].0, 17);
    assert_eq!(arena[idx3].0, 17);
    arena[idx2].0 = 18;
    let mut iter = arena.iter();
    assert!(iter.next() == Some((idx1, &T(42))));
    assert!(iter.next() == Some((idx2, &T(18))));
    assert!(iter.next() == Some((idx3, &T(17))));
    assert!(iter.next().is_none());
}

#[test]
fn alloc_many() {
    let mut arena = Arena::<_, u32>::new();
    let span = arena.alloc_many([10, 20, 30]);
    assert_eq!(format!("{:?}", span), "IdxSpan::<i32, u32>(0..3)");
    assert_eq!(&arena[span], &[10, 20, 30]);
    arena[span][1] = 21;
    let mut iter = arena.iter();
    assert_eq!(iter.next().map(|(i, v)| (i.into_raw(), v)), Some((0, &10)));
    assert_eq!(iter.next().map(|(i, v)| (i.into_raw(), v)), Some((1, &21)));
    assert_eq!(iter.next().map(|(i, v)| (i.into_raw(), v)), Some((2, &30)));
    assert_eq!(iter.next(), None);
}

#[test]
fn alloc_many_twice() {
    let mut arena = Arena::<_, u32>::new();
    let span1 = arena.alloc_many([1, 2]);
    let span2 = arena.alloc_many([3, 4]);
    assert_eq!(format!("{:?}", span1), "IdxSpan::<i32, u32>(0..2)");
    assert_eq!(format!("{:?}", span2), "IdxSpan::<i32, u32>(2..4)");
    assert_eq!(&arena[span1], &[1, 2]);
    assert_eq!(&arena[span2], &[3, 4]);
}

#[test]
fn idx_span_len_and_is_empty() {
    let normal = IdxSpan::<u32, u32>::new(2..5);
    assert_eq!(normal.len(), 3);
    assert!(!normal.is_empty());

    let empty = IdxSpan::<u32, u32>::new(4..4);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    // A reversed range is treated as empty and must not underflow.
    // (A literal `5..2` would be rejected by a lint, so build the range from variables.)
    let (start, end) = (5, 2);
    let reversed = IdxSpan::<u32, u32>::new(start..end);
    assert_eq!(reversed.len(), 0);
    assert!(reversed.is_empty());
}

#[test]
fn try_alloc_many_rolls_back_on_full() {
    // A `u8` arena can hold at most `u8::MAX` (255) elements.
    let mut arena = Arena::<u32, u8>::new();
    for i in 0..254 {
        arena.alloc(i);
    }
    assert_eq!(arena.len(), 254);

    // Only one slot is left, so allocating three elements must fail and leave
    // the arena untouched (no orphaned, unreferenced elements).
    let span = arena.try_alloc_many([1000, 1001, 1002]);
    assert_eq!(span, None);
    assert_eq!(arena.len(), 254);
    assert!(arena.values().copied().eq(0..254));
}

#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
mod no_ub {
    use super::*;
    use std::panic;

    fn id_from<T: Id>() {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let max = <T as Id>::MAX;
            if max == usize::MAX {
                panic!("arithmetic overflow");
            }
            let _ = <T as Id>::from_usize(max + 1);
        }));
        assert!(result.is_err());
        let _ = <T as Id>::from_usize(<T as Id>::MAX);
    }
    mktest!(id_from_u8, id_from::<u8>());
    mktest!(id_from_u16, id_from::<u16>());
    mktest!(id_from_u32, id_from::<u32>());
    mktest!(id_from_u64, id_from::<u64>());
    mktest!(id_from_usize, id_from::<usize>());
    mktest!(id_from_nz_u8, id_from::<NonZero<u8>>());
    mktest!(id_from_nz_u16, id_from::<NonZero<u16>>());
    mktest!(id_from_nz_u32, id_from::<NonZero<u32>>());
    mktest!(id_from_nz_u64, id_from::<NonZero<u64>>());
    mktest!(id_from_nz_usize, id_from::<NonZero<usize>>());
}
