const std = @import("std");
const math = @import("std").math;
const checkType = @import("./types.zig").checkType;

const TensorError = error{
    WrongNumberOfIndices,
    IndexOutOfBounds,
};

pub fn _Tensor(comptime Type: type) type {
    const isComplex = if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) true else false;

    return struct {
        _shape: []usize,
        _strides: []usize,
        _requires_grad: bool,
        _isComplex: bool = isComplex,

        _T: []Type,
        grad: []Type,

        const Self = @This();

        pub fn init(shape: []const usize, data: ?[]const Type, requires_grad: bool) !Self {
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }

            const tShape = try std.heap.page_allocator.alloc(usize, shape.len);
            for (shape, 0..) |d, i| {
                tShape[i] = d;
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
                ._shape = tShape,
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

        pub fn is_complex(self: Self) bool {
            return self._isComplex;
        }

        // TODO: Should be visualized in the terminal
        pub fn print(self: Self) !void {
            for (self._T) |t| {
                std.debug.print("{d} ", .{t});
            }
            std.debug.print("\n", .{});
        }

        pub fn new_tensor(
            self: Self,
            shape: []const usize,
            data: []const Type,
        ) !_Tensor(Type) {
            return _Tensor(Type).init(shape, data, self._requires_grad);
        }

        pub fn new_full(self: Self, shape: []const usize, data: Type) !_Tensor(Type) {
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

        pub fn new_empty(self: Self, shape: []const usize) !_Tensor(Type) {
            return _Tensor(Type).init(shape, null, self._requires_grad);
        }

        pub fn new_ones(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.new_full(shape, 1);
        }

        pub fn new_zeros(self: Self, shape: []const usize) !_Tensor(Type) {
            return self.new_full(shape, 0);
        }

        pub fn element_size(self: Self) usize {
            _ = self;
            return @sizeOf(Type);
        }
    };
}

// Test
test "init" {
    std.debug.print("Initialize the tensor\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    _ = tensor;
}

test "set and get" {
    std.debug.print("Set and get\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);

    try tensor.set(&[_]usize{ 1, 2, 0 }, 3.14);
    const result = try tensor.get(&[_]usize{ 1, 2, 0 });
    try std.testing.expect(3.14 == result);
}

test "Complex" {
    std.debug.print("Complex\n", .{});
    const f32Tensor = _Tensor(math.Complex(f32));
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.is_complex());
}

test "is not Complex" {
    std.debug.print("is not Complex\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(!tensor.is_complex());
}

test "index out of bounds" {
    std.debug.print("Index out of bounds\n", .{});
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);

    try std.testing.expectError(TensorError.IndexOutOfBounds, tensor.set(&[_]usize{ 3, 2, 0 }, 3.14));
}

test "get dimemsion" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.dim() == 3);
    std.debug.print("Expected 3, got {}\n", .{tensor.dim()});
}

test "new_tensor" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_tensor(&[_]usize{3}, &[_]f32{ 1, 2, 3 });
    try new_tensor.print();
}

test "new_full" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_full(&[_]usize{3}, 3);
    try new_tensor.print();
}

test "new_empty" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_empty(&[_]usize{3});
    try new_tensor.print();
}

test "new_ones" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_ones(&[_]usize{3});
    try new_tensor.print();
}

test "new_zeros" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_zeros(&[_]usize{3});
    try new_tensor.print();
}

test "element_size" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.element_size() == @sizeOf(f32));
    std.debug.print("Expected 4, got {}\n", .{tensor.element_size()});
}
