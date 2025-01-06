namespace NeuralNetworkFromScratch.Loss;

public class MeanSquaredError
{
	public double Loss(double y, double yHat) => (yHat - y) * (yHat - y) / 2.0;

	public double Derivative(double y, double yHat) => yHat - y;
}
