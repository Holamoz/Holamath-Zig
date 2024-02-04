const std = @import("std");

/// Useable types of _Tensor
pub const tensorTypes = [_]type{
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    f16,
    f32,
    f64,
    std.math.Complex(f64),
    std.math.Complex(f32),
    std.math.Complex(f16),
    bool,
};
