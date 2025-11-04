using System;
using System.Runtime.CompilerServices;

namespace NNImage.Models;

/// <summary>
/// Fixed-size 256-bit mask optimized for palette sizes up to 256.
/// Backed by four 64-bit words to allow fast bitwise ops and popcount.
/// </summary>
public struct BitMask256
{
    public ulong W0; // bits 0..63
    public ulong W1; // bits 64..127
    public ulong W2; // bits 128..191
    public ulong W3; // bits 192..255

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BitMask256 Full(int paletteSize)
    {
        if (paletteSize >= 256) return new BitMask256 { W0 = ulong.MaxValue, W1 = ulong.MaxValue, W2 = ulong.MaxValue, W3 = ulong.MaxValue };
        var mask = new BitMask256 { W0 = ulong.MaxValue, W1 = ulong.MaxValue, W2 = ulong.MaxValue, W3 = ulong.MaxValue };
        // Clear the tail bits above paletteSize
        int extra = 256 - paletteSize;
        if (extra > 0)
        {
            for (int i = 0; i < extra; i++)
                mask.UnsetBit(255 - i);
        }
        return mask;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BitMask256 FromSingle(int index)
    {
        var m = default(BitMask256);
        m.SetBit(index);
        return m;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetBit(int idx)
    {
        if (idx < 64) W0 |= 1UL << idx;
        else if (idx < 128) W1 |= 1UL << (idx - 64);
        else if (idx < 192) W2 |= 1UL << (idx - 128);
        else W3 |= 1UL << (idx - 192);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UnsetBit(int idx)
    {
        if (idx < 64) W0 &= ~(1UL << idx);
        else if (idx < 128) W1 &= ~(1UL << (idx - 64));
        else if (idx < 192) W2 &= ~(1UL << (idx - 128));
        else W3 &= ~(1UL << (idx - 192));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TestBit(int idx)
    {
        if (idx < 64) return (W0 & (1UL << idx)) != 0;
        if (idx < 128) return (W1 & (1UL << (idx - 64))) != 0;
        if (idx < 192) return (W2 & (1UL << (idx - 128))) != 0;
        return (W3 & (1UL << (idx - 192))) != 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int PopCount()
    {
        return System.Numerics.BitOperations.PopCount(W0)
             + System.Numerics.BitOperations.PopCount(W1)
             + System.Numerics.BitOperations.PopCount(W2)
             + System.Numerics.BitOperations.PopCount(W3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static BitMask256 And(BitMask256 a, BitMask256 b)
    {
        return new BitMask256
        {
            W0 = a.W0 & b.W0,
            W1 = a.W1 & b.W1,
            W2 = a.W2 & b.W2,
            W3 = a.W3 & b.W3,
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AndInPlace(BitMask256 other)
    {
        W0 &= other.W0;
        W1 &= other.W1;
        W2 &= other.W2;
        W3 &= other.W3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Equals(BitMask256 other)
    {
        return W0 == other.W0 && W1 == other.W1 && W2 == other.W2 && W3 == other.W3;
    }

    public bool IsZero => W0 == 0 && W1 == 0 && W2 == 0 && W3 == 0;
}