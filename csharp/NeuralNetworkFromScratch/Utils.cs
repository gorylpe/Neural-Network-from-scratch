namespace NeuralNetworkFromScratch;

public static class Utils
{
	public static void ConsoleWriteYAndYHat(double[] Y, double[][] Yhat) =>
		ConsoleWriteYAndYHat(Y, Yhat.Select(x => x[0]).ToArray());

	public static void ConsoleWriteYAndYHat(double[] Y, double[] Yhat) =>
		Console.WriteLine(string.Join("; ", Y.Zip(Yhat).Select(x => $"{x.Item1} -> {x.Item2:F2}")));


	public static double CalculateAndConsoleWriteAccuracy(double[] Y, double[][] Yhat) =>
		CalculateAndConsoleWriteAccuracy(Y, Yhat.Select(x => x[0]).ToArray());

	public static double CalculateAndConsoleWriteAccuracy(double[] Y, double[] Yhat)
	{
		var errors = Yhat.Select((yhat, i) => (yhat > 0.5 ? 1 : 0) == Y[i]).Count(x => !x);
		var accuracy = 1.0 - 1.0 * errors / Y.Length;
		Console.WriteLine($"Accuracy: {accuracy:P}, {errors} errors out of {Y.Length} examples");
		return accuracy;
	}
}
