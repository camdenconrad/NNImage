namespace NNImage.Models;

public enum Direction
{
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest
}

public static class DirectionExtensions
{
    public static (int dx, int dy) GetOffset(this Direction direction)
    {
        return direction switch
        {
            Direction.North => (0, -1),
            Direction.NorthEast => (1, -1),
            Direction.East => (1, 0),
            Direction.SouthEast => (1, 1),
            Direction.South => (0, 1),
            Direction.SouthWest => (-1, 1),
            Direction.West => (-1, 0),
            Direction.NorthWest => (-1, -1),
            _ => (0, 0)
        };
    }

    public static Direction Opposite(this Direction direction)
    {
        return direction switch
        {
            Direction.North => Direction.South,
            Direction.NorthEast => Direction.SouthWest,
            Direction.East => Direction.West,
            Direction.SouthEast => Direction.NorthWest,
            Direction.South => Direction.North,
            Direction.SouthWest => Direction.NorthEast,
            Direction.West => Direction.East,
            Direction.NorthWest => Direction.SouthEast,
            _ => direction
        };
    }

    public static Direction[] AllDirections { get; } = new[]
    {
        Direction.North,
        Direction.NorthEast,
        Direction.East,
        Direction.SouthEast,
        Direction.South,
        Direction.SouthWest,
        Direction.West,
        Direction.NorthWest
    };
}
