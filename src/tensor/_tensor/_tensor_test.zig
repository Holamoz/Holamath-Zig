const std = @import("std");
const math = std.math;
const _Tensor = @import("./_tensor.zig")._Tensor;
const TensorError = @import("./_tensor.zig").TensorError;

test "Init and Deinit" {
    std.debug.print("Init and Deinit \n", .{});
    const i8Tensor = _Tensor(i8);
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer t.deinit();

    try std.testing.expect(t.isComplex() == false);
    try std.testing.expect(t.elementSize() == @sizeOf(i8));
    try std.testing.expect(t._shape[0] == 2 and t._shape[1] == 3);
    try std.testing.expect(t._T[0] == 1 and t._T[1] == 2 and t._T[2] == 3 and t._T[3] == 4 and t._T[4] == 5 and t._T[5] == 6);
}

test "_Tensor.set and _Tensor.get" {
    std.debug.print("Set and get\n", .{});
    const i8Tensor = _Tensor(i8);
    var tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer tensor.deinit();

    try tensor.set(&[_]usize{ 1, 2 }, 3);
    const result = try tensor.get(&[_]usize{ 1, 2 });
    try std.testing.expect(3 == result);
}

test "_Tensor Complex" {
    std.debug.print("Complex\n", .{});
    const c64Tensor = _Tensor(std.math.Complex(f32));
    var t = try c64Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, null, false);
    defer t.deinit();
    try std.testing.expect(t.isComplex());
}

test "_Tensor is not Complex" {
    std.debug.print("Not Complex\n", .{});
    const i8Tensor = _Tensor(i8);
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer t.deinit();
    try std.testing.expect(!t.isComplex());
}

test "_Tensor expected wrong: index out of bounds" {
    std.debug.print("Index out of bounds\n", .{});
    const i8Tensor = _Tensor(i8);
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer t.deinit();

    try std.testing.expectError(TensorError.IndexOutOfBounds, t.set(&[_]usize{ 3, 3 }, 3));
}

test "_Tensor.dim()" {
    const i8Tensor = _Tensor(i8);
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer t.deinit();
    try std.testing.expect(t.dim() == 2);
    std.debug.print("Expected 2, got {}\n", .{t.dim()});
}

test "_Tensor.newTensor()" {
    const i8Tensor = _Tensor(i8);
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 3 }, &[_]i8{ 1, 2, 3, 4, 5, 6 }, false);
    defer t.deinit();
    var new_tensor = try t.newTensor(&[_]usize{3}, &[_]i8{ 1, 2, 3 });
    defer new_tensor.deinit();
    new_tensor.print();
}

test "_Tensor.newFull()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(std.testing.allocator, &[_]usize{ 3, 3, 3 }, null, false);
    defer tensor.deinit();
    var new_tensor = try tensor.newFull(&[_]usize{3}, 3);
    defer new_tensor.deinit();
    new_tensor.print();
}

test "_Tensor.newEmpty()" {
    const u8Tensor = _Tensor(u8);
    var tensor: u8Tensor = try u8Tensor.init(std.testing.allocator, &[_]usize{ 3, 3, 3 }, null, false);
    defer tensor.deinit();
    var new_tensor = try tensor.newEmpty(&[_]usize{3});
    defer new_tensor.deinit();
    new_tensor.print();
}

test "_Tensor.newOnes()" {
    const u8Tensor = _Tensor(u8);
    var tensor: u8Tensor = try u8Tensor.init(std.testing.allocator, &[_]usize{ 3, 3, 3 }, null, false);
    defer tensor.deinit();
    var new_tensor = try tensor.newOnes(&[_]usize{3});
    defer new_tensor.deinit();
    new_tensor.print();
}

test "_Tensor.newZeros()" {
    const u8Tensor = _Tensor(u8);
    var tensor: u8Tensor = try u8Tensor.init(std.testing.allocator, &[_]usize{ 3, 3, 3 }, null, false);
    defer tensor.deinit();
    var new_tensor = try tensor.newZeros(&[_]usize{3});
    defer new_tensor.deinit();
    new_tensor.print();
}

test "_Tensor.elementSize()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(std.testing.allocator, &[_]usize{ 3, 3, 3 }, null, false);
    defer tensor.deinit();
    try std.testing.expect(tensor.elementSize() == @sizeOf(f32));
    std.debug.print("Expected 4, got {}\n", .{tensor.elementSize()});
}

test "_Tensor.clamp_()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    _ = try tensor.clamp_(-3, 3);
    tensor.print();
    try std.testing.expect(tensor._T[0] == -3 and tensor._T[1] == 2 and tensor._T[2] == 3);
}

test "_Tensor.clamp()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    const clamped = try tensor.clamp(-3, 3);
    clamped.print();
    defer clamped.deinit();
    try std.testing.expect((clamped._T[0] == -3 and clamped._T[1] == 2 and clamped._T[2] == 3));
}

test "_Tensor.clone()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    defer tensor.deinit();
    var cloned = try tensor.clone();
    defer cloned.deinit();
    cloned.print();
    try std.testing.expect((cloned.dim() == tensor.dim()));
    try std.testing.expectEqual(cloned._T[0], tensor._T[0]);
}

test "_Tensor.copy_()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{2}, &[_]i8{ -5, 8 }, false);
    defer tensor.deinit();
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 1, 3 }, &[_]i8{ 1, 2, 3 }, false);
    defer t.deinit();
    _ = try t.copy_(tensor);
    std.debug.print("copy_: ", .{});
    t.print();
    try std.testing.expect((t.dim() != tensor.dim()));
    try std.testing.expectEqual(t._T[0], tensor._T[0]);
}

test "_Tensor.equal() - not equal" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{2}, &[_]i8{ -5, 8 }, false);
    defer tensor.deinit();
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{ 1, 3 }, &[_]i8{ 1, 2, 3 }, false);
    defer t.deinit();
    const e = t.equal(tensor);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == false);
}

test "_Tensor.equal() - equal" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{2}, &[_]i8{ -5, 8 }, false);
    defer tensor.deinit();
    var t = try i8Tensor.init(std.testing.allocator, &[_]usize{2}, &[_]i8{ -5, 8 }, false);
    defer t.deinit();
    const e = t.equal(tensor);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == true);
}

test "_Tensor.reshape()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer tensor.deinit();
    var reshaped = try tensor.reshape(&[_]usize{ 1, 4 });
    defer reshaped.deinit();
    reshaped.print();
    try std.testing.expectEqual(reshaped._T[0], tensor._T[0]);
    try std.testing.expect(reshaped._shape[0] == 1);
    try std.testing.expect(reshaped._shape[1] == 4);
}

test "_Tensor.reshape_as()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer tensor.deinit();
    var target: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 1, 4 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer target.deinit();
    var reshaped = try tensor.reshape_as(target);
    defer reshaped.deinit();
    reshaped.print();
    try std.testing.expectEqual(reshaped._T[0], tensor._T[0]);
    try std.testing.expect(reshaped._shape[0] == 1);
    try std.testing.expect(reshaped._shape[1] == 4);
}

test "_Tensor.resize_()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer tensor.deinit();
    var resized = try tensor.resize_(&[_]usize{ 1, 2 });
    defer resized.deinit();
    resized.print();
    std.debug.print("resize._T: {*}", .{tensor._T.ptr});
    try std.testing.expectEqual(resized._T[0], tensor._T[0]);
    try std.testing.expect(resized._shape[0] == 1);
    try std.testing.expect(resized._shape[1] == 2);
    try std.testing.expect(resized._T.len == 1 * 2);
}

test "_Tensor.view()" {
    const i8Tensor = _Tensor(i8);
    var tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer tensor.deinit();
    var viewed = try tensor.view(&[_]usize{ 1, 4 });
    defer viewed.deinit();
    viewed.print();
    try std.testing.expect(viewed._shape[0] == 1);
    try std.testing.expect(viewed._shape[1] == 4);
}

test "_Tensor.view() - Expect Tensor shape incompatible error" {
    const i8Tensor = _Tensor(i8);
    var tensor = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer tensor.deinit();
    try std.testing.expectError(TensorError.TensorIncompatibleShape, tensor.view(&[_]usize{ 1, 2 }));
}

test "_Tensor.view_as()" {
    const i8Tensor = _Tensor(i8);
    var t1 = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer t1.deinit();
    var t2 = try i8Tensor.init(std.testing.allocator, &[_]usize{ 1, 4 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer t2.deinit();
    var viewed = try t1.view_as(t2);
    defer viewed.deinit();
    viewed.print();
    try std.testing.expect(viewed._shape[0] == 1);
    try std.testing.expect(viewed._shape[1] == 4);
}

test "_Tensor.zero_()" {
    const i8Tensor = _Tensor(i8);
    var t1 = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    defer t1.deinit();
    var t2 = t1.zero_();
    t2.print();
    std.debug.print("_Tensor.zero_()", .{});
    try std.testing.expect(t1._T[0] == 0);
    try std.testing.expect(t1._T[2] == 0);
    try std.testing.expect(t2._T[1] == 0);
    try std.testing.expect(t2._T[3] == 0);
}

test "_Tensor.round()_" {
    const f32Tensor = _Tensor(f32);
    var t1 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, 2.4, 3.7, 4.1 }, false);
    defer t1.deinit();
    try t1.round_();
    t1.print();
}

test "_Tensor.round_() - with complex" {
    const c64Tensor = _Tensor(math.Complex(f64));
    const c64v1 = math.Complex(f64).init(1.2, 2.4);
    const c64v2 = math.Complex(f64).init(3.7, 4.1);
    var t1 = try c64Tensor.init(std.testing.allocator, &[_]usize{ 1, 2 }, &[_]std.math.Complex(f64){ c64v1, c64v2 }, false);
    defer t1.deinit();
    try t1.round_();
    t1.print();
}

test "_Tensor.round()" {
    const f32Tensor = _Tensor(f32);
    var t1 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, 2.4, 3.7, 4.1 }, false);
    defer t1.deinit();
    const rounded = try t1.round();
    defer rounded.deinit();
    rounded.print();
    try std.testing.expect(rounded._T[0] == 1);
    try std.testing.expect(rounded._T[1] == 2);
    try std.testing.expect(rounded._T[2] == 4);
    try std.testing.expect(rounded._T[3] == 4);
}

test "_Tensor.abs_()" {
    const f32Tensor = _Tensor(f32);
    var t1 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, -2.4, 3.7, -4.1 }, false);
    defer t1.deinit();
    try t1.abs_();
    t1.print();
    try std.testing.expect(t1._T[0] == 1.2);
    try std.testing.expect(t1._T[1] == 2.4);
    try std.testing.expect(t1._T[2] == 3.7);
    try std.testing.expect(t1._T[3] == 4.1);
}

test "_Tensor.abs_() - with int" {
    const i8Tensor = _Tensor(i8);
    var t1 = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, -2, 3, -4 }, false);
    defer t1.deinit();
    try t1.abs_();
    t1.print();
    try std.testing.expect(t1._T[0] == 1);
    try std.testing.expect(t1._T[1] == 2);
    try std.testing.expect(t1._T[2] == 3);
    try std.testing.expect(t1._T[3] == 4);
}

test "_Tensor.abs_() - with complex" {
    const c64Tensor = _Tensor(math.Complex(f64));
    const c64v1 = math.Complex(f64).init(3, 4);
    const c64v2 = math.Complex(f64).init(-5, 12);
    var t1 = try c64Tensor.init(std.testing.allocator, &[_]usize{ 1, 2 }, &[_]std.math.Complex(f64){ c64v1, c64v2 }, false);
    defer t1.deinit();
    try t1.abs_();
    t1.print();
    try std.testing.expect(t1._T[0].re == 5);
    try std.testing.expect(t1._T[0].im == 0);
    try std.testing.expect(t1._T[1].re == 13);
    try std.testing.expect(t1._T[1].im == 0);
}

test "_Tensor.abs()" {
    const i8Tensor = _Tensor(i8);
    var t1 = try i8Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]i8{ 1, -2, 3, -4 }, false);
    defer t1.deinit();
    const abs = try t1.abs();
    defer abs.deinit();
    abs.print();
    try std.testing.expect(abs._T[0] == 1);
    try std.testing.expect(abs._T[1] == 2);
    try std.testing.expect(abs._T[2] == 3);
    try std.testing.expect(abs._T[3] == 4);
}

test "_Tensor.abs() - with complex" {
    const c64Tensor = _Tensor(math.Complex(f64));
    const c64v1 = math.Complex(f64).init(3, 4);
    const c64v2 = math.Complex(f64).init(-5, 12);
    var t1 = try c64Tensor.init(std.testing.allocator, &[_]usize{ 1, 2 }, &[_]std.math.Complex(f64){ c64v1, c64v2 }, false);
    defer t1.deinit();
    const abs = try t1.abs();
    defer abs.deinit();
    abs.print();
    try std.testing.expect(abs._T[0].re == 5);
    try std.testing.expect(abs._T[0].im == 0);
    try std.testing.expect(abs._T[1].re == 13);
    try std.testing.expect(abs._T[1].im == 0);
}

test "_Tensor.add_()" {
    const f32Tensor = _Tensor(f32);
    var t1 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, -2.4, 3.7, -4.1 }, false);
    defer t1.deinit();
    var t2 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, -2.4, 3.7, -4.1 }, false);
    defer t2.deinit();
    try t1.add_(t2);
    t1.print();
    try std.testing.expect(t1._T[0] == 2.4);
    try std.testing.expect(t1._T[1] == -4.8);
    try std.testing.expect(t1._T[2] == 7.4);
    try std.testing.expect(t1._T[3] == -8.2);
}

test "_Tensor.add()" {
    const f32Tensor = _Tensor(f32);
    var t1 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, -2.4, 3.7, -4.1 }, false);
    defer t1.deinit();
    var t2 = try f32Tensor.init(std.testing.allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1.2, -2.4, 3.7, -4.1 }, false);
    defer t2.deinit();
    const sum = try t1.add(t2);
    defer sum.deinit();
    sum.print();
    try std.testing.expect(sum._T[0] == 2.4);
    try std.testing.expect(sum._T[1] == -4.8);
    try std.testing.expect(sum._T[2] == 7.4);
    try std.testing.expect(sum._T[3] == -8.2);
}

test "_Tesor.add() - with complex" {
    const c64Tensor = _Tensor(math.Complex(f64));
    const c64v1 = math.Complex(f64).init(3, 4);
    const c64v2 = math.Complex(f64).init(-5, 12);
    var t1 = try c64Tensor.init(std.testing.allocator, &[_]usize{ 1, 2 }, &[_]std.math.Complex(f64){ c64v1, c64v2 }, false);
    defer t1.deinit();
    const sum = try t1.add(t1);
    defer sum.deinit();
    sum.print();
    try std.testing.expect(sum._T[0].re == 6);
    try std.testing.expect(sum._T[0].im == 8);
    try std.testing.expect(sum._T[1].re == -10);
    try std.testing.expect(sum._T[1].im == 24);
}
