namespace NeuralNetworkFromScratch.Regularizers;

public interface IKernelRegularizer
{
	double Loss(double[][]           weights);
	void   Regularize(double[][] dwRegularized, double[][] weights);
}
