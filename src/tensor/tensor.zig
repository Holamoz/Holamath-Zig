const std = @import("std");

const _Tensor = @import("./_tensor/_tensor.zig")._Tensor;
const TensorError = @import("./_tensor/_tensor.zig").TensorError;

pub fn tensor(comptime Type: type, comptime shape: []const usize, comptime data: ?[]const Type, comptime requires_grad: bool) !_Tensor(Type) {
    return _Tensor(Type).init(shape, data, requires_grad);
}

test "tensor" {
    const t = try tensor(f32, &[_]usize{ 2, 3 }, null, false);
    std.debug.print("tensor\n", .{});
    try std.testing.expect(t.is_complex() == false);
    try std.testing.expect(t.element_size() == @sizeOf(f32));
}
