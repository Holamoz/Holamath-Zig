const std = @import("std");
const math = @import("std").math;

pub const TensorError = error{
    WrongNumberOfIndices,
    IndexOutOfBounds,
    TypeNotSupported,
};

pub fn _Tensor(comptime Type: type) type {
    const typeIsComplex = if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) true else false;

    return struct {
        _shape: []const usize,
        _strides: []usize,
        _requires_grad: bool,
        _isComplex: bool = typeIsComplex,

        _T: []Type,
        grad: []Type,

        const Self = @This();

        pub fn init(shape: []const usize, data: ?[]const Type, requires_grad: bool) !Self {
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }

            const tensorData = try std.heap.page_allocator.alloc(Type, size);
            if (data) |d| {
                std.mem.copy(Type, tensorData, d);
            }

            const strides = try std.heap.page_allocator.alloc(usize, shape.len);
            strides[0] = 1;
            for (1..shape.len) |i| {
                strides[i] = strides[i - 1] * shape[i - 1];
            }

            return Self{
                ._T = tensorData,
                ._shape = shape,
                ._strides = strides,
                .grad = try std.heap.page_allocator.alloc(Type, size), // need to implment the backward etc.
                ._requires_grad = requires_grad,
            };
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
            return _Tensor(Type).init(shape, data, self._requires_grad);
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
            return _Tensor(Type).init(shape, tdata, self._requires_grad);
        }

        pub fn newEmpty(self: Self, shape: []const usize) !_Tensor(Type) {
            return _Tensor(Type).init(shape, null, self._requires_grad);
        }

        pub fn newOnes(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.newFull(shape, 1);
        }

        pub fn newZeros(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.newFull(shape, 0);
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
            return _Tensor(Type).init(self._shape, self._T, self._requires_grad);
        }

        pub fn copy_(self: Self, src: Self) !_Tensor(Type) {
            const len = @min(self._T.len, src._T.len);
            std.mem.copy(Type, self._T[0..len], src._T[0..len]);
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
            return _Tensor(Type).init(shape, self._T, self._requires_grad);
        }

        pub fn reshape_as(self: Self, other: Self) !_Tensor(Type) {
            return _Tensor(Type).init(other._shape, self._T, self._requires_grad);
        }
    };
}

// Test
test "_Tensor.init" {
    std.debug.print("Initialize the tensor\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    _ = tensor;
}

test "_Tensor.set and _Tensor.get" {
    std.debug.print("Set and get\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);

    try tensor.set(&[_]usize{ 1, 2, 0 }, 3.14);
    const result = try tensor.get(&[_]usize{ 1, 2, 0 });
    try std.testing.expect(3.14 == result);
}

test "_Tensor Complex" {
    std.debug.print("Complex\n", .{});
    const f32Tensor = _Tensor(math.Complex(f32));
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.isComplex());
}

test "_Tensor is not Complex" {
    std.debug.print("is not Complex\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(!tensor.isComplex());
}

test "_Tensor expected wrong: index out of bounds" {
    std.debug.print("Index out of bounds\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);

    try std.testing.expectError(TensorError.IndexOutOfBounds, tensor.set(&[_]usize{ 3, 2, 0 }, 3.14));
}

test "_Tensor.dim()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.dim() == 3);
    std.debug.print("Expected 3, got {}\n", .{tensor.dim()});
}

test "_Tensor.new_tensor()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.newTensor(&[_]usize{3}, &[_]f32{ 1, 2, 3 });
    new_tensor.print();
}

test "_Tensor.new_full()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.newFull(&[_]usize{3}, 3);
    new_tensor.print();
}

test "_Tensor.new_empty()" {
    const u8Tensor = _Tensor(u8);
    var tensor: u8Tensor = try u8Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.newEmpty(&[_]usize{3});
    new_tensor.print();
}

test "_Tensor.new_ones()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.newOnes(&[_]usize{3});
    new_tensor.print();
}

test "_Tensor.new_zeros()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.newZeros(&[_]usize{3});
    new_tensor.print();
}

test "_Tensor.element_size()" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.elementSize() == @sizeOf(f32));
    std.debug.print("Expected 4, got {}\n", .{tensor.elementSize()});
}

test "_Tensor.clamp_()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    _ = try tensor.clamp_(-3, 3);
    tensor.print();
}

test "_Tensor.clamp()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    const clamped = try tensor.clamp(-3, 3);
    clamped.print();
}

test "_Tensor.clone()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{3}, &[_]i8{ -5, 2, 8 }, false);
    var cloned = try tensor.clone();
    cloned.print();
    try std.testing.expect((cloned.dim() == tensor.dim()));
    try std.testing.expectEqual(cloned._T[0], tensor._T[0]);
}

test "_Tensor.copy_()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{2}, &[_]i8{ -5, 8 }, false);
    var t = try i8Tensor.init(&[_]usize{ 1, 3 }, &[_]i8{ 1, 2, 3 }, false);
    _ = try t.copy_(tensor);
    std.debug.print("copy_: ", .{});
    t.print();
    try std.testing.expect((t.dim() != tensor.dim()));
    try std.testing.expectEqual(t._T[0], tensor._T[0]);
}

test "_Tensor.equal() - not equal" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{2}, &[_]i8{ -5, 8 }, false);
    var t = try i8Tensor.init(&[_]usize{ 1, 3 }, &[_]i8{ 1, 2, 3 }, false);
    var e = t.equal(tensor);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == false);
}

test "_Tensor.equal() - equal" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{2}, &[_]i8{ -5, 8 }, false);
    var t = try i8Tensor.init(&[_]usize{2}, &[_]i8{ -5, 8 }, false);
    var e = t.equal(tensor);
    std.debug.print("equal: {}\n", .{e});
    try std.testing.expect(e == true);
}

test "_Tensor.reshape()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    var reshaped = try tensor.reshape(&[_]usize{ 1, 4 });
    reshaped.print();
    try std.testing.expectEqual(reshaped._T[0], tensor._T[0]);
    try std.testing.expect(reshaped._shape[0] == 1);
    try std.testing.expect(reshaped._shape[1] == 4);
}

test "_Tensor.reshape_as()" {
    const i8Tensor = _Tensor(i8);
    var tensor: i8Tensor = try i8Tensor.init(&[_]usize{ 2, 2 }, &[_]i8{ 1, 2, 3, 4 }, false);
    var target: i8Tensor = try i8Tensor.init(&[_]usize{ 1, 4 }, &[_]i8{ 1, 2, 3, 4 }, false);
    var reshaped = try tensor.reshape_as(target);
    reshaped.print();
    try std.testing.expectEqual(reshaped._T[0], tensor._T[0]);
    try std.testing.expect(reshaped._shape[0] == 1);
    try std.testing.expect(reshaped._shape[1] == 4);
}
