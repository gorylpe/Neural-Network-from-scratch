namespace NeuralNetworkFromScratch;

public static class TestData
{
	public static (double[][], double[]) LoadCoffeeData(int examples = 400, int seed = 2)
	{
		var rng = new Random(seed);
		var Y = new double[examples];
		var X = new double[examples][];

		for (var i = 0; i < examples; i++)
		{
			var d = rng.NextDouble() * 4 + 11.5; // Roasting duration
			var t = rng.NextDouble() * (275 - 160) + 160; // Temperature
			var y = -3.0 / (260 - 175) * t + 21;
			X[i] = [t, d];
			Y[i] = t is > 175 and < 260 && d is > 12 and < 15 && d <= y ? 1 : 0;
		}

		return (X, Y);
	}

	public static (double[][], double[]) LoadLogisticData()
	{
		double[][] X = [[0], [1], [2], [3], [4], [5]];
		double[] Y = [0, 0, 0, 1, 1, 1];

		return (X, Y);
	}

	public static (double[][], double[]) LoadLinearData()
	{
		return ([[1], [2]], [300, 500]);
	}
}
