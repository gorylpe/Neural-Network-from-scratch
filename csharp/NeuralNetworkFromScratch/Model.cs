using System.Numerics;
using NeuralNetworkFromScratch.Layers;

namespace NeuralNetworkFromScratch;

public class Model
{
	private readonly List<Dense> _layers;

	public Model(List<Dense> layers)
	{
		VerifyLayers(layers);
		_layers = layers;
	}

	private void VerifyLayers(List<Dense> layers)
	{
		int? previousLayerUnits = null;
		foreach (var layer in layers)
		{
			if (previousLayerUnits != null && layer.GetInputSize() != previousLayerUnits)
				throw new ArgumentException(
					$"The input size of the layer {layer.GetInputSize()} does not match the previous layer's output size {previousLayerUnits}.");

			previousLayerUnits = layer.GetUnits();
		}
	}
	
	public Dense GetLayer(int index) => _layers[index];

	public double[] Predict(double[] x) => _layers.Aggregate(x, (current, layer) => layer.Forward(current));
	public double[][] Predict(double[][] X) => _layers.Aggregate(X, (current, layer) => layer.Forward(current));
}
