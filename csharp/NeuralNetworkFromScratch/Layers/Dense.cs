using System.Numerics;

namespace NeuralNetworkFromScratch.Layers;

public class Dense : ILayer
{
	private readonly ActivationType _activationType;

	private double[][] _weights;
	private double[]   _biases;


	public Dense(int units, int inputSize, ActivationType activationType, double[][]? weights = null,
		double[]? biases = null)
	{
		_activationType = activationType;
		_weights = new double[inputSize][];
		for (var i = 0; i < inputSize; i++)
			_weights[i] = new double[units];
		_biases = new double[units];

		if (weights != null && biases != null)
			SetWeightsAndBiases(weights, biases);
	}

	public int GetInputSize() => _weights.Length;
	public int GetUnits()     => _weights[0].Length;

	public double[][] GetWeights() => _weights;
	public double[]   GetBiases()  => _biases;

	public void SetWeightsAndBiases(double[][] weights, double[] biases)
	{
		if (weights.Length != GetInputSize())
			throw new ArgumentException("Weight size must be equal to input size. (number of features)");
		if (weights[0].Length != GetUnits())
			throw new ArgumentException("Single weight size must be equal to number of units.");
		_weights = weights;
		_biases = biases;
	}

	public double[] Forward(double[] x) => Activation.Forward(_activationType, x, _weights, _biases);

	public double[][] Forward(double[][] X) => X.Select(Forward).ToArray();
}
