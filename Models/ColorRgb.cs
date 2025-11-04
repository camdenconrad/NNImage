using System;

namespace NNImage.Models;

public readonly struct ColorRgb : IEquatable<ColorRgb>
{
    public byte R { get; }
    public byte G { get; }
    public byte B { get; }

    public ColorRgb(byte r, byte g, byte b)
    {
        R = r;
        G = g;
        B = b;
    }

    public bool Equals(ColorRgb other)
    {
        return R == other.R && G == other.G && B == other.B;
    }

    public override bool Equals(object? obj)
    {
        return obj is ColorRgb other && Equals(other);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(R, G, B);
    }

    public static bool operator ==(ColorRgb left, ColorRgb right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(ColorRgb left, ColorRgb right)
    {
        return !left.Equals(right);
    }

    public override string ToString()
    {
        return $"RGB({R}, {G}, {B})";
    }

    public uint ToUInt32()
    {
        return (uint)((255 << 24) | (R << 16) | (G << 8) | B);
    }

    public static ColorRgb FromUInt32(uint value)
    {
        var r = (byte)((value >> 16) & 0xFF);
        var g = (byte)((value >> 8) & 0xFF);
        var b = (byte)(value & 0xFF);
        return new ColorRgb(r, g, b);
    }
}
