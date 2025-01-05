using System.Numerics;

namespace NeuralNetworkFromScratch.Layers;

public interface ILayer
{
	double[]   Forward(double[] x);
	double[][] Forward(double[][] X);
}
