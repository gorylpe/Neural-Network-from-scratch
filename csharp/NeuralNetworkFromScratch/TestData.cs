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
			var t = rng.NextDouble() * (285 - 150) + 150; // Temperature
			var y = -3.0 / (260 - 175) * t + 21;
			X[i] = [t, d];
			Y[i] = t is > 175 and < 260 && d is > 12 and < 15 && d <= y ? 1 : 0;
		}

		return (X, Y);
	}
}
