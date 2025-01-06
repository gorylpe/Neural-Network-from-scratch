namespace NeuralNetworkFromScratch.Loss;

public class BinaryCrossEntropy
{
	public double Loss(double y, double yHat)
	{
		return y switch
		{
			0.0 => -Math.Log(yHat),
			1.0 => -Math.Log(1.0 - yHat),
			_ => throw new ArgumentException("y must be 0 or 1.")
		};
	}

	public double Derivative(double y, double aOut)
	{
		return y switch
		{
			0.0 => 1.0 / (1.0 - aOut),
			1.0 => 1.0 / aOut,
			_ => throw new ArgumentException("y must be 0 or 1.")
		};
	}

	public double DerivativeLogits(double y, double zOut) => zOut - y;
}
