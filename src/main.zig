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

test "Tensor.empty()" {
    const t = try Tensor.empty(u8, &[_]usize{ 2, 3 }, false);
    std.debug.print("empty\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.emptyLike()" {
    const testTensor = try Tensor.tensor(u8, &[_]usize{ 2, 3 }, null, false);
    const t = try Tensor.emptyLike(u8, testTensor);
    std.debug.print("emptyLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.full()" {
    const t = try Tensor.full(u8, &[_]usize{ 2, 3 }, 3, false);
    std.debug.print("full\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.fullLike()" {
    const testTensor = try Tensor.tensor(u8, &[_]usize{ 2, 3 }, null, false);
    const t = try Tensor.fullLike(u8, testTensor, 3);
    std.debug.print("fullLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.clamp()" {
    var tensor = try Tensor.tensor(i8, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    const t = try Tensor.clamp(i8, tensor, 0, 1);
    std.debug.print("clamp: ", .{});
    t.print();
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t._T[0] == 0);
}

test "Tensor.clone()" {
    var tensor = try Tensor.tensor(i8, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    const t = try Tensor.clone(i8, tensor);
    std.debug.print("clone: ", .{});
    t.print();
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t._T[0] == -5);
}

test "Tensor.equal()" {
    var tensor = try Tensor.tensor(i8, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    var t = try Tensor.tensor(i8, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    var e = Tensor.equal(i8, tensor, t);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == true);
}
