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

        pub fn set(self: Self, indices: []const usize, value: Type) !void {
            const index = try self.calculateIndex(indices);
            self._T[index] = value;
        }

        pub fn get(self: Self, indices: []const usize) !Type {
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
            if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) {
                for (self._T) |t| {
                    std.debug.print("({}, {}) ", .{ t.re, t.im });
                }
            } else {
                for (self._T) |t| {
                    std.debug.print("{d} ", .{t});
                }
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
            return self.view(shape);
        }

        pub fn reshape_as(self: Self, other: Self) !_Tensor(Type) {
            return self.view_as(other);
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
                for (self._T) |*d| {
                    d.*.re = @round(d.re);
                    d.*.im = @round(d.im);
                }
            } else {
                for (self._T) |*d| {
                    d.* = @round(d.*);
                }
            }
        }

        pub fn round(self: Self) !_Tensor(Type) {
            var t = try self.clone();
            try t.round_();
            return t;
        }

        pub fn abs_(self: Self) !void {
            if (Type == math.Complex(f16) or Type == math.Complex(f32) or Type == math.Complex(f64)) {
                for (self._T) |*d| {
                    d.*.re = @abs(d.re);
                    d.*.im = @abs(d.im);
                }
            } else if (Type == f16 or Type == f32 or Type == f64) {
                for (self._T) |*d| {
                    d.* = @abs(d.*);
                }
            } else if (Type == i8 or Type == i16 or Type == i32 or Type == i64) {
                for (self._T) |*d| {
                    d.* = @intCast(@abs(d.*));
                }
            } else {
                @compileError("abs_ not implemented for type " ++ @typeName(Type));
            }
        }

        pub fn abs(self: Self) !_Tensor(Type) {
            var t = try self.clone();
            try t.abs_();
            return t;
        }

        pub fn add_(self: *Self, other: Self) !void {
            // ensure that both tensors have the same shape
            if (self._shape.len != other._shape.len) {
                return TensorError.TensorIncompatibleShape;
            } else {
                for (self._shape, 0..self._shape.len) |s, i| {
                    if (s != other._shape[i]) {
                        return TensorError.TensorIncompatibleShape;
                    }
                }
            }

            for (self._T, 0..self._T.len) |*d, i| {
                d.* = d.* + other._T[i];
            }
        }

        pub fn add(self: Self, other: Self) !_Tensor(Type) {
            var t = try self.clone();
            try t.add_(other);
            return t;
        }
    };
}
