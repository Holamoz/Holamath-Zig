const std = @import("std");

const _Tensor = @import("./_tensor/_tensor.zig")._Tensor;
const TensorError = @import("./_tensor/_tensor.zig").TensorError;

pub fn tensor(comptime Type: type, comptime shape: []const usize, comptime data: ?[]const Type, comptime requires_grad: bool) !_Tensor(Type) {
    return _Tensor(Type).init(shape, data, requires_grad);
}

pub fn ones(comptime Type: type, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, shape, null, requires_grad);
    return t.new_ones(shape);
}
