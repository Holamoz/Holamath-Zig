const std = @import("std");
const testing = std.testing;

const Tensor = @import("tensor/tensor.zig");

pub fn holamath() []const u8 {
    return "Hola, math!";
}

// Tests
const _tensor_test = @import("tensor/_tensor/_tensor_test.zig");
test {
    _ = _tensor_test;
}

test "holamath" {
    std.debug.print("holamath\n", .{});
    try testing.expect(std.mem.eql(u8, holamath(), "Hola, math!"));
}

test "Tensor.tensor()" {
    const t = try Tensor.tensor(u8, std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer t.deinit();
    std.debug.print("tensor\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
}

test "Tensor.ones()" {
    const t = try Tensor.ones(u8, std.testing.allocator, &[_]usize{ 2, 3 }, false);
    defer t.deinit();
    std.debug.print("ones\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.zeros()" {
    const t = try Tensor.zeros(u8, std.testing.allocator, &[_]usize{ 2, 3 }, false);
    defer t.deinit();
    std.debug.print("zeros\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.onesLike()" {
    const testTensor = try Tensor.tensor(u8, std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer testTensor.deinit();
    const t = try Tensor.onesLike(u8, testTensor);
    defer t.deinit();
    std.debug.print("onesLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.zerosLike()" {
    const testTensor = try Tensor.tensor(u8, std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer testTensor.deinit();
    const t = try Tensor.zerosLike(u8, testTensor);
    defer t.deinit();
    std.debug.print("zerosLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.empty()" {
    const t = try Tensor.empty(u8, std.testing.allocator, &[_]usize{ 2, 3 }, false);
    defer t.deinit();
    std.debug.print("empty\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.emptyLike()" {
    const testTensor = try Tensor.tensor(u8, std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer testTensor.deinit();
    const t = try Tensor.emptyLike(u8, testTensor);
    defer t.deinit();
    std.debug.print("emptyLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.full()" {
    const t = try Tensor.full(u8, std.testing.allocator, &[_]usize{ 2, 3 }, 3, false);
    defer t.deinit();
    std.debug.print("full\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.fullLike()" {
    const testTensor = try Tensor.tensor(u8, std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer testTensor.deinit();
    const t = try Tensor.fullLike(u8, testTensor, 3);
    defer t.deinit();
    std.debug.print("fullLikes\n", .{});
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(u8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.clamp()" {
    const tensor = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const t = try Tensor.clamp(i8, tensor, 0, 1);
    defer t.deinit();
    std.debug.print("clamp: ", .{});
    t.print();
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t._T[0] == 0);
}

test "Tensor.clone()" {
    const tensor = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const t = try Tensor.clone(i8, tensor);
    defer t.deinit();
    std.debug.print("clone: ", .{});
    t.print();
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t._T[0] == -5);
}

test "Tensor.equal()" {
    const tensor = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const t = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer t.deinit();
    const e = Tensor.equal(i8, tensor, t);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == true);
}

test "Tensor.reshape()" {
    const tensor = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const t = try Tensor.reshape(i8, tensor, &[_]usize{ 3, 1 });
    defer t.deinit();
    std.debug.print("reshape: ", .{});
    t.print();
    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t._T[0] == -5);
    try std.testing.expect(t._shape[0] == 3);
    try std.testing.expect(t._shape[1] == 1);
}

test "Tensor.round()" {
    const tensor = try Tensor.tensor(f32, std.testing.allocator, &[_]usize{ 1, 3 }, &[_]f32{ 1.2, 2.3, 3.3 }, false);
    defer tensor.deinit();
    const t = try Tensor.round(f32, tensor);
    defer t.deinit();
    t.print();
    try std.testing.expect(t._T[0] == 1);
    try std.testing.expect(t._T[1] == 2);
    try std.testing.expect(t._T[2] == 3);
}

test "Tenor.abs()" {
    const tensor = try Tensor.tensor(i8, std.testing.allocator, &[_]usize{ 1, 3 }, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const t = try Tensor.abs(i8, tensor);
    defer t.deinit();
    t.print();
    try std.testing.expect(t._T[0] == 5);
    try std.testing.expect(t._T[1] == 2);
    try std.testing.expect(t._T[2] == 8);
}
