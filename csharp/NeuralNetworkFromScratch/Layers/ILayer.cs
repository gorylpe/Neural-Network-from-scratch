using System.Numerics;

namespace NeuralNetworkFromScratch.Layers;

public interface ILayer
{
	void InitializeWeightsForTraining(Random? random = null);

	(double RegularizationLoss, double[] Activations) Forward(double[] x);
}
