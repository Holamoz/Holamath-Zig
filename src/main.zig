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
    const t = try Tensor.tensor(f32, &[_]usize{ 2, 3 }, null, false);
    std.debug.print("tensor\n", .{});
    try std.testing.expect(t.is_complex() == false);
    try std.testing.expect(t.element_size() == @sizeOf(f32));
}

test "Tensor.ones()" {
    const t = try Tensor.ones(f32, &[_]usize{ 2, 3 }, false);
    std.debug.print("ones\n", .{});
    try std.testing.expect(t.is_complex() == false);
    try std.testing.expect(t.element_size() == @sizeOf(f32));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}

test "Tensor.zeros()" {
    const t = try Tensor.zeros(f32, &[_]usize{ 2, 3 }, false);
    std.debug.print("zeros\n", .{});
    try std.testing.expect(t.is_complex() == false);
    try std.testing.expect(t.element_size() == @sizeOf(f32));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
}
