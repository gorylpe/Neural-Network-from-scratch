namespace NeuralNetworkFromScratch.Loss;

public interface ILossCalc
{
	double Loss(double       y, double yHat);
	double Derivative(double y, double yHat);
}
