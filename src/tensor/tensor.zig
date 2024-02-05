const std = @import("std");

const _Tensor = @import("./_tensor/_tensor.zig")._Tensor;
const TensorError = @import("./_tensor/_tensor.zig").TensorError;

pub fn tensor(comptime Type: type, comptime shape: []const usize, comptime data: ?[]const Type, comptime requires_grad: bool) !_Tensor(Type) {
    return _Tensor(Type).init(shape, data, requires_grad);
}

pub fn ones(comptime Type: type, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, shape, null, requires_grad);
    return t.newOnes(shape);
}

pub fn zeros(comptime Type: type, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, shape, null, requires_grad);
    return t.newZeros(shape);
}

pub fn empty(comptime Type: type, comptime shape: []const usize, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, shape, null, requires_grad);
    return t.newEmpty(shape);
}

pub fn full(comptime Type: type, comptime shape: []const usize, comptime data: Type, comptime requires_grad: bool) !_Tensor(Type) {
    const t = try tensor(Type, shape, null, requires_grad);
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
