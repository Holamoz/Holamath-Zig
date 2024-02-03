const std = @import("std");

const TensorError = error{ WrongNumberOfIndices, IndexOutOfBounds };

pub fn _Tensor(comptime Type: type) type {
    return struct {
        dim: usize,
        shape: []usize,
        strides: []usize,
        requires_grad: bool,

        T: []Type,
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
                .T = tensorData,
                .shape = tShape,
                .dim = shape.len,
                .strides = strides,
                .grad = try std.heap.page_allocator.alloc(Type, size), // need to implment the backward etc.
                .requires_grad = requires_grad,
            };
        }

        fn set(self: Self, indices: []const usize, value: Type) !void {
            const index = try self.calculateIndex(indices);
            self.T[index] = value;
        }

        fn get(self: Self, indices: []const usize) !Type {
            const index = try self.calculateIndex(indices);
            return self.T[index];
        }

        fn calculateIndex(self: Self, indices: []const usize) !usize {
            if (indices.len != self.shape.len) {
                return TensorError.WrongNumberOfIndices;
            }
            var index: usize = 0;
            var stride: usize = 1;
            for (indices, 0..) |idx, i| {
                if (idx >= self.shape[i]) {
                    return TensorError.IndexOutOfBounds;
                }
                index += idx * stride;
                stride *= self.shape[i];
            }
            return index;
        }

        pub fn getDim(self: Self) usize {
            return self.dim;
        }

        pub fn print(self: Self) !void {
            for (self.T) |t| {
                std.debug.print("{d} ", .{t});
            }
            std.debug.print("\n", .{});
        }

        pub fn new_tensor(self: Self, shape: []const usize, data: []const Type, require_grad: bool) !_Tensor(Type) {
            _ = self;
            return _Tensor(Type).init(shape, data, require_grad);
        }

        pub fn new_full(self: Self, shape: []const usize, data: Type, require_grad: bool) !_Tensor(Type) {
            _ = self;
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }
            const tdata = try std.heap.page_allocator.alloc(Type, size);
            for (0..tdata.len) |i| {
                tdata[i] = data;
            }
            return _Tensor(Type).init(shape, tdata, require_grad);
        }

        pub fn new_empty(self: Self, shape: []const usize, require_grad: bool) !_Tensor(Type) {
            _ = self;
            return _Tensor(Type).init(shape, null, require_grad);
        }
    };
}

// Test
test "init" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    _ = tensor;
}

test "set and get" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);

    try tensor.set(&[_]usize{ 1, 2, 0 }, 3.14);
    const result = try tensor.get(&[_]usize{ 1, 2, 0 });
    try std.testing.expect(3.14 == result);
}

// test "index out of bounds" {
//     const f32Tensor = Tensor(f32);
//     var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 });

//     try tensor.set(&[_]usize{ 3, 2, 0 }, 3.14); // This should fail
// }

test "get dimemsion" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    try std.testing.expect(tensor.getDim() == 3);
}

test "new_tensor" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_tensor(&[_]usize{3}, &[_]f32{ 1, 2, 3 }, false);
    try new_tensor.print();
}

test "new_full" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_full(&[_]usize{3}, 3, false);
    try new_tensor.print();
}

test "new_empty" {
    const f32Tensor = _Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 }, null, false);
    var new_tensor = try tensor.new_empty(&[_]usize{3}, false);
    try new_tensor.print();
}
