const std = @import("std");
const testing = std.testing;

pub fn holamath() []const u8 {
    return "Hola, math!";
}

test "holamath" {
    try testing.expect(std.mem.eql(u8, holamath(), "Hola, math!"));
}
