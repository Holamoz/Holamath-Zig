const std = @import("std");

const _Tensor = @import("./_tensor/_tensor.zig")._Tensor;
const TensorError = @import("./_tensor/_tensor.zig").TensorError;

pub fn tensor(comptime Type: type, allocator: std.mem.Allocator, comptime shape: []const usize, comptime data: ?[]const Type, comptime requires_grad: bool) !_Tensor(Type) {
    return _Tensor(Type).init(allocator, shape, data, requires_grad);
}

pub fn ones(comptime Type: type, allocator: std.mem.Allocator, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, allocator, shape, null, requires_grad);
    defer t.deinit();
    return t.newOnes(shape);
}

pub fn zeros(comptime Type: type, allocator: std.mem.Allocator, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, allocator, shape, null, requires_grad);
    return t.zero_();
}

pub fn empty(comptime Type: type, allocator: std.mem.Allocator, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, allocator, shape, null, requires_grad);
    defer t.deinit();
    return t.newEmpty(shape);
}

pub fn full(comptime Type: type, allocator: std.mem.Allocator, comptime shape: []const usize, comptime data: Type, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, allocator, shape, null, requires_grad);
    defer t.deinit();
    return t.newFull(shape, data);
}

pub fn onesLike(comptime Type: type, T: _Tensor(Type)) !_Tensor(Type) {
    return T.newOnes(T._shape);
}

pub fn zerosLike(comptime Type: type, T: _Tensor(Type)) !_Tensor(Type) {
    return T.newZeros(T._shape);
}

pub fn emptyLike(comptime Type: type, T: _Tensor(Type)) !_Tensor(Type) {
    return T.newEmpty(T._shape);
}

pub fn fullLike(comptime Type: type, T: _Tensor(Type), data: Type) !_Tensor(Type) {
    return T.newFull(T._shape, data);
}

pub fn clamp(comptime Type: type, T: _Tensor(Type), min: Type, max: Type) !_Tensor(Type) {
    return T.clamp(min, max);
}

pub fn clone(comptime Type: type, T: _Tensor(Type)) !_Tensor(Type) {
    return T.clone();
}

pub fn equal(comptime Type: type, Input: _Tensor(Type), Other: _Tensor(Type)) bool {
    return Input.equal(Other);
}

pub fn reshape(comptime Type: type, Input: _Tensor(Type), shape: []const usize) !_Tensor(Type) {
    return Input.reshape(shape);
}
