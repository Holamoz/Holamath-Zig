const std = @import("std");
const testing = std.testing;

const Tensor = @import("tensor/tensor.zig");

pub fn holamath() []const u8 {
    return "Hola, math!";
}

test "holamath" {
    std.debug.print("holamath\n", .{});
    try testing.expect(std.mem.eql(u8, holamath(), "Hola, math!"));
}

test "Tensor.tensor()" {
    const t = try Tensor.tensor(u8, &[_]usize{ 2, 3 }, null, false);
    std.debug.print("tensor\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
}

test "Tensor.ones()" {
    const t = try Tensor.ones(u8, &[_]usize{ 2, 3 }, false);
    std.debug.print("ones\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.zeros()" {
    const t = try Tensor.zeros(u8, &[_]usize{ 2, 3 }, false);
    std.debug.print("zeros\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.onesLike()" {
    const testTensor = try Tensor.tensor(u8, &[_]usize{ 2, 3 }, null, false);
    const t = try Tensor.onesLike(u8, testTensor);
    std.debug.print("onesLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.zerosLike()" {
    const testTensor = try Tensor.tensor(u8, &[_]usize{ 2, 3 }, null, false);
    const t = try Tensor.zerosLike(u8, testTensor);
    std.debug.print("zerosLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}
