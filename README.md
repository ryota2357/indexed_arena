# indexed_arena

[![Crates.io](https://img.shields.io/crates/v/indexed_arena.svg)](https://crates.io/crates/indexed_arena)
[![Documentation](https://docs.rs/indexed_arena/badge.svg)](https://docs.rs/indexed_arena)

A simple, index-based arena without deletion.

This crate is inspired by [la-arena](https://crates.io/crates/la-arena), which is used in [rust-analyzer](https://github.com/rust-lang/rust-analyzer).

This crate's [Arena\<T, I\>](https://docs.rs/indexed_arena/latest/indexed_arena/struct.Arena.html) provides similar functionality and an API to [la-arena's Arena\<T\>](https://docs.rs/la-arena/latest/la_arena/struct.Arena.html), but with an abstracted internal index representation and several additional features.

## Example

### Simple Usage

```rust
use indexed_arena::{Arena, Idx};

let mut arena: Arena<&str, u32> = Arena::new();
let idx: Idx<&str, u32> = arena.alloc("hello");
assert_eq!(arena[idx], "hello");
```

`Arena` is defined as `struct Arena<T, I: Id> { ... }`.
Here, `T` represents the type of elements stored in the arena, and `I` specifies the index type used by the arena.

In this example, we use `u32`, a built-in numeric type that implements the `Id` trait provided by this library.

The choice of `I` determines the maximum number of elements the arena can hold.
For instance, using `u32` allows up to 4,294,967,295 elements, while using `u16` would limit the arena to 65,535 elements.

### Building a simple syntax tree with niche optimization

```rust
use core::{num::NonZero, mem::size_of_val};
use indexed_arena::{Arena, Idx};

type ExprId = Idx<Expr, NonZero<u32>>;

#[derive(Debug, PartialEq, Eq)]
enum Expr {
    Number(i32),
    Add(ExprId, ExprId),
}

let mut arena: Arena<Expr, NonZero<u32>> = Arena::new();
let id1 = arena.alloc(Expr::Number(1));
let id2 = arena.alloc(Expr::Number(2));
let add_expr = arena.alloc(Expr::Add(id1, id2));

assert_eq!(arena[add_expr], Expr::Add(id1, id2));

// Option<ExprId> is efficiently represented thanks to the niche of NonZero<u32>.
let maybe_expr: Option<ExprId> = None;
assert_eq!(size_of_val(&maybe_expr), size_of::<ExprId>());
```

###  Distinguishing arenas with the same element type using wrapped index types

If you have two arenas that store the same element type `T` and use the same underlying index type (for example, `u16`),
their indices (`Idx<T, u16>`) would be indistinguishable, potentially leading to mix-ups.
To resolve this, you can wrap the numeric index type in newtypes so that the indices are differentiated at the type level.

```rust
use indexed_arena::{Arena, Idx, Id};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
struct TypeA(u16);
impl Id for TypeA {
    const MAX: usize = <u16 as Id>::MAX as usize;
    fn from_usize(idx: usize) -> Self {
        TypeA(<u16 as Id>::from_usize(idx))
    }
    fn into_usize(self) -> usize {
        <u16 as Id>::into_usize(self.0)
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
struct TypeB(u16);
impl Id for TypeB {
    const MAX: usize = <u16 as Id>::MAX as usize;
    fn from_usize(idx: usize) -> Self {
        TypeB(<u16 as Id>::from_usize(idx))
    }
    fn into_usize(self) -> usize {
        <u16 as Id>::into_usize(self.0)
    }
}

let mut arena_a: Arena<&'static str, TypeA> = Arena::new();
let mut arena_b: Arena<&'static str, TypeB> = Arena::new();

let id_a = arena_a.alloc("from arena A");
let id_b = arena_b.alloc("from arena B");

// Note that id_a and id_b have different types:
//   id_a: Idx<&'static str, TypeA>
//   id_b: Idx<&'static str, TypeB>
// This prevents mixing up indices between arenas.
assert_eq!((arena_a[id_a]), "from arena A");
assert_eq!((arena_b[id_b]), "from arena B");
```

## License

This crate is licensed under the MIT license.
