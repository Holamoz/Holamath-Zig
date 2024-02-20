const std = @import("std");
const math = @import("std").math;

pub const TensorError = error{
    WrongNumberOfIndices,
    IndexOutOfBounds,
    TypeNotSupported,
    TensorIncompatibleShape,
};

pub fn _Tensor(comptime Type: type) type {
    const typeIsComplex = if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) true else false;

    return struct {
        _shape: []usize,
        _strides: []usize,
        _requires_grad: bool,
        _isComplex: bool = typeIsComplex,

        _T: []Type,
        grad: []Type,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(
            allocator: std.mem.Allocator,
            shape: []const usize,
            data: ?[]const Type,
            requires_grad: bool,
        ) !Self {
            var self = Self.generate(allocator, requires_grad);
            try self.ensureContainers(shape, data);
            return self;
        }

        pub fn generate(allocator: std.mem.Allocator, requires_grad: bool) Self {
            return Self{
                ._T = &[_]Type{},
                ._shape = &[_]usize{},
                ._strides = &[_]usize{},
                .grad = &[_]Type{},
                ._requires_grad = requires_grad,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self._shape.ptr[0..self._shape.len]);
            self.allocator.free(self._strides.ptr[0..self._strides.len]);
            self.allocator.free(self._T.ptr[0..self._T.len]);
            self.allocator.free(self.grad.ptr[0..self.grad.len]);
        }

        fn ensureContainers(self: *Self, newShape: []const usize, newData: ?[]const Type) std.mem.Allocator.Error!void {
            const oldShapePtrs = self._shape.ptr[0..self._shape.len];
            if (self.allocator.resize(oldShapePtrs, newShape.len)) {
                @memcpy(self._shape, newShape);
            } else {
                const newShapeMemory = try self.allocator.alloc(usize, newShape.len);
                @memcpy(newShapeMemory, newShape);
                self.allocator.free(oldShapePtrs);
                self._shape = newShapeMemory;
            }

            const newStrides = try self.allocator.alloc(usize, newShape.len);
            defer self.allocator.free(newStrides);
            newStrides[0] = 1;
            for (1..newShape.len) |i| {
                newStrides[i] = newStrides[i - 1] * newShape[i - 1];
            }
            if (self.allocator.resize(self._strides, newStrides.len)) {
                @memcpy(self._strides, newStrides);
            } else {
                const newStridesMemory = try self.allocator.alloc(usize, newStrides.len);
                @memcpy(newStridesMemory, newStrides);
                self.allocator.free(self._strides);
                self._strides = newStridesMemory;
            }

            if (newData) |d| {
                var size: usize = 1;
                for (newShape) |s| {
                    size *= s;
                }
                const oldDataPtrs = self._T.ptr[0..self._T.len];
                if (self.allocator.resize(oldDataPtrs, size)) {
                    @memcpy(self._T, d);
                } else {
                    const newDataMemory = try self.allocator.alloc(Type, size);
                    @memcpy(newDataMemory, d[0..size]);
                    self.allocator.free(oldDataPtrs);
                    self._T = newDataMemory;
                }
            }
        }

        fn set(self: Self, indices: []const usize, value: Type) !void {
            const index = try self.calculateIndex(indices);
            self._T[index] = value;
        }

        fn get(self: Self, indices: []const usize) !Type {
            const index = try self.calculateIndex(indices);
            return self._T[index];
        }

        fn calculateIndex(self: Self, indices: []const usize) !usize {
            if (indices.len != self._shape.len) {
                return TensorError.WrongNumberOfIndices;
            }
            var index: usize = 0;
            var stride: usize = 1;
            for (indices, 0..) |idx, i| {
                if (idx >= self._shape[i]) {
                    return TensorError.IndexOutOfBounds;
                }
                index += idx * stride;
                stride *= self._shape[i];
            }
            return index;
        }

        pub fn dim(self: Self) usize {
            return self._shape.len;
        }

        pub fn isComplex(self: Self) bool {
            return self._isComplex;
        }

        pub fn getShape(self: Self) []const usize {
            return self._shape;
        }

        // TODO: Should be visualized in the terminal
        pub fn print(self: Self) void {
            for (self._T) |t| {
                std.debug.print("{d} ", .{t});
            }
            std.debug.print("\n", .{});
        }

        pub fn newTensor(
            self: Self,
            shape: []const usize,
            data: []const Type,
        ) !_Tensor(Type) {
            return _Tensor(Type).init(self.allocator, shape, data, self._requires_grad);
        }

        pub fn newFull(self: Self, shape: []const usize, data: Type) !_Tensor(Type) {
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }
            const tdata = try std.heap.page_allocator.alloc(Type, size);
            for (0..tdata.len) |i| {
                tdata[i] = data;
            }
            return _Tensor(Type).init(self.allocator, shape, tdata, self._requires_grad);
        }

        pub fn newEmpty(self: Self, shape: []const usize) !_Tensor(Type) {
            return _Tensor(Type).init(self.allocator, shape, null, self._requires_grad);
        }

        pub fn newOnes(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.newFull(shape, 1);
        }

        pub fn newZeros(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.newFull(shape, 0);
        }

        pub fn zero_(self: Self) _Tensor(Type) {
            @memset(self._T, 0);
            return self;
        }

        pub fn elementSize(self: Self) usize {
            _ = self;
            return @sizeOf(Type);
        }

        pub fn clamp_(self: Self, min: Type, max: Type) !_Tensor(Type) {
            for (self._T) |*t| {
                t.* = @min(@max(t.*, min), max);
            }
            return self;
        }

        pub fn clamp(self: Self, min: Type, max: Type) !_Tensor(Type) {
            var t = try self.newTensor(self._shape, self._T);
            return try t.clamp_(min, max);
        }

        pub fn clone(self: Self) !_Tensor(Type) {
            return _Tensor(Type).init(self.allocator, self._shape, self._T, self._requires_grad);
        }

        pub fn copy_(self: Self, src: Self) !_Tensor(Type) {
            const len = @min(self._T.len, src._T.len);
            @memcpy(self._T[0..len], src._T[0..len]);
            return self;
        }

        pub fn equal(self: Self, other: Self) bool {
            if (self._shape.len != other._shape.len) {
                return false;
            } else if (self._T.len != other._T.len) {
                return false;
            } else {
                for (self._T, other._T) |t1, t2| {
                    if (t1 != t2) {
                        return false;
                    }
                }
                return true;
            }
        }

        pub fn reshape(self: Self, shape: []const usize) !_Tensor(Type) {
            return _Tensor(Type).init(self.allocator, shape, self._T, self._requires_grad);
        }

        pub fn reshape_as(self: Self, other: Self) !_Tensor(Type) {
            return _Tensor(Type).init(self.allocator, other._shape, self._T, self._requires_grad);
        }

        pub fn resize_(self: *Self, shape: []const usize) !_Tensor(Type) {
            try self.ensureContainers(shape, null);

            return _Tensor(Type).init(self.allocator, shape, self._T, self._requires_grad);
        }

        pub fn view(self: Self, shape: []const usize) !_Tensor(Type) {
            var newSize: usize = 1;
            for (shape) |s| {
                newSize *= s;
            }
            var curSize: usize = 1;
            for (self._shape) |s| {
                curSize *= s;
            }

            if (newSize != curSize) {
                return TensorError.TensorIncompatibleShape;
            }
            return _Tensor(Type).init(std.testing.allocator, shape, self._T, self._requires_grad);
        }

        pub fn view_as(self: Self, other: _Tensor(Type)) !_Tensor(Type) {
            return self.view(other._shape);
        }

        pub fn round_(self: Self) !void {
            if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) {
                std.debug.print("is complex {}", .{self.isComplex()});
                for (self._T) |*d| {
                    d.*.real = std.math.round(d.real);
                    d.*.img = std.math.round(d.img);
                }
            } else {
                for (self._T) |*d| {
                    d.* = @round(d.*);
                }
            }
        }
    };
}

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
}
