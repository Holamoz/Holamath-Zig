const std = @import("std");

const TensorError = error{ WrongNumberOfIndices, IndexOutOfBounds };

fn Tensor(comptime Type: type) type {
    return struct {
        data: []Type,
        dim: usize,
        shape: []usize,

        const Self = @This();

        fn init(shape: []const usize) !Self {
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }
            const ternsorShape = try std.heap.page_allocator.alloc(usize, shape.len);
            const tensorData = try std.heap.page_allocator.alloc(Type, size);
            const tensorDim = shape.len;

            for (shape, 0..) |d, i| {
                ternsorShape[i] = d;
            }
            return Self{
                .data = tensorData,
                .shape = ternsorShape,
                .dim = tensorDim,
            };
        }

        /// Sets the value in the tensor at the specified indices.
        ///
        /// - Parameters:
        ///   - self: The tensor.
        ///   - indices: The list of indices.
        ///   - value: The value to be set.
        /// - Returns: void or an error.
        fn set(self: Self, indices: []const usize, value: Type) !void {
            const index = try self.calculateIndex(indices);
            self.data[index] = value;
        }

        /// Retrieves a value from the tensor using the specified indices.
        ///
        /// - Parameters
        ///
        ///   - `self` - A reference to the `Tensor` object.
        ///   - `indices` - An immutable slice of unsigned integers representing the indices.
        ///
        /// - Returns: The value of type `Type` retrieved from the tensor.
        fn get(self: Self, indices: []const usize) !Type {
            const index = try self.calculateIndex(indices);
            return self.data[index];
        }

        /// Calculates the linear index of a multi-dimensional array based on the given indices.
        ///
        /// - Parameters
        ///   - `self` - A reference to the `Tensor` object.
        ///   - `indices` - An immutable slice of unsigned integers representing the indices.
        ///
        /// - Returns: The calculated linear index of type `usize`.
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

        fn getDim(self: Self) usize {
            return self.dim;
        }
    };
}

// Test
test "init" {
    const f32Tensor = Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 });
    _ = tensor;
}

test "set and get" {
    const f32Tensor = Tensor(f32);
    var tensor: f32Tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 });

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
    const f32Tensor = Tensor(f32);
    var tensor = try f32Tensor.init(&[_]usize{ 3, 3, 3 });
    try std.testing.expect(tensor.getDim() == 3);
}
