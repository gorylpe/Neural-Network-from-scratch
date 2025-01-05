namespace NeuralNetworkFromScratch;

public static class ArrayExtensions
{
	public static string PrettyString<T>(this T[] array) => $"[{string.Join(", ", array)}]";
}
