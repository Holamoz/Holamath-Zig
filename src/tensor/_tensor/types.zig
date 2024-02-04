const std = @import("std");

/// Useable types of _Tensor
pub const tensorTypes = [_][]const u8{
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "i64",
    "u64",
    "f16",
    "f32",
    "f64",
    "std.math.Complex(f64)",
    "std.math.Complex(f32)",
    "std.math.Complex(f16)",
    "bool",
};

pub fn checkType(comptime T: type) bool {
    inline for (tensorTypes) |typeString| {
        if (std.mem.eql(u8, @typeName(T), typeString)) {
            return true;
        }
    }
    return false;
}

test "checkTypeWithTypeString" {
    try std.testing.expect(checkType(f32));
}
