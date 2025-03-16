use indexed_arena::{Arena, Id};
use std::{fmt::Debug, num::NonZero};

macro_rules! test {
    ($name:ident, $expr:expr) => {
        #[test]
        fn $name() {
            $expr
        }
    };
}
fn construct<T: Id + Debug>(idx_str: &str) {
    let mut arena = Arena::<String, T>::new();
    assert_eq!(arena.len(), 0);
    assert!(arena.is_empty());
    let idx = arena.alloc("foo".to_string());
    assert_eq!(arena.len(), 1);
    assert!(!arena.is_empty());
    assert_eq!(format!("{:?}", idx), idx_str);
}
test!(construct_u8, construct::<u8>("Idx::<String, u8>(0)"));
test!(construct_u16, construct::<u16>("Idx::<String, u16>(0)"));
test!(construct_u32, construct::<u32>("Idx::<String, u32>(0)"));
test!(construct_u64, construct::<u64>("Idx::<String, u64>(0)"));
test!(construct_usize, construct::<usize>("Idx::<String, usize>(0)"));
test!(construct_nz_u8, construct::<NonZero<u8>>("Idx::<String, NonZero<u8>>(1)"));
test!(construct_nz_u16, construct::<NonZero<u16>>("Idx::<String, NonZero<u16>>(1)"));
test!(construct_nz_u32, construct::<NonZero<u32>>("Idx::<String, NonZero<u32>>(1)"));
test!(construct_nz_u64, construct::<NonZero<u64>>("Idx::<String, NonZero<u64>>(1)"));
test!(construct_nz_usize, construct::<NonZero<usize>>("Idx::<String, NonZero<usize>>(1)"));

#[test]
fn alloc_get_iter() {
    #[derive(PartialEq)]
    struct T(u32);
    let mut arena = Arena::<_, u32>::new();
    let idx1 = arena.alloc(T(42));
    assert_eq!(format!("{:?}", idx1), "Idx::<T, u32>(0)");
    let idx2 = arena.alloc(T(17));
    assert_eq!(format!("{:?}", idx2), "Idx::<T, u32>(1)");
    let idx3 = arena.alloc(T(17));
    assert_eq!(format!("{:?}", idx3), "Idx::<T, u32>(2)");
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
