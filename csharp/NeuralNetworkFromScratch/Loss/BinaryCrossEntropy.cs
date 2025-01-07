namespace NeuralNetworkFromScratch.Loss;

public class BinaryCrossEntropy(bool fromLogits = false) : ILossCalc
{
	public double Loss(double y, double yHat)
	{
		yHat = ClampYHatForNumericalStability(yHat);
		return y switch
		{
			0.0 => -Math.Log(yHat),
			1.0 => -Math.Log(1.0 - yHat),
			_   => throw new ArgumentException("y must be 0 or 1.")
		};
	}

	public double Derivative(double y, double yHat)
	{
		if (fromLogits)
			return DerivativeLogits(y, yHat);

		yHat = ClampYHatForNumericalStability(yHat);
		return y switch
		{
			0.0 => 1.0 / (1.0 - yHat),
			1.0 => 1.0 / yHat,
			_   => throw new ArgumentException("y must be 0 or 1.")
		};
	}

	private double DerivativeLogits(double y, double yHat) => yHat - y;

	private static double ClampYHatForNumericalStability(double yHat) => Math.Clamp(yHat, 1e-15, 1.0 - 1e-15);
}
