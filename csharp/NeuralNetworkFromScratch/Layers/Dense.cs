using System.Numerics;

namespace NeuralNetworkFromScratch.Layers;

public class Dense : ILayer
{
	private readonly ActivationType _activationType;

	private double[][] _weights;
	private double[] _biases;

	public Dense(int units, int inputSize, ActivationType activationType, double[][]? weights = null,
		double[]? biases = null)
	{
		_activationType = activationType;
		_weights = new double[inputSize][];
		for (var i = 0; i < inputSize; i++)
			_weights[i] = new double[units];
		_biases = new double[units];

		if (weights != null)
			SetWeights(weights);
		if (biases != null)
			SetBiases(biases);
	}

	public void InitializeWeightsForTraining()
	{
		var inputSize = _weights.Length;
		var units = _weights[0].Length;

		var useXavier = _activationType == ActivationType.Sigmoid;

		var random = new Random();
		var scale = useXavier
			? Math.Sqrt(1.0 / inputSize) // Xavier Initialization
			: Math.Sqrt(2.0 / inputSize); // He Initialization
		var weights = new double[inputSize][];
		for (var i = 0; i < inputSize; i++)
		{
			weights[i] = new double[units];
			for (var j = 0; j < units; j++)
				weights[i][j] = random.NextDouble() * 2 - 1 * scale; // Randomized in range [-scale, scale]
		}

		SetWeights(weights);
	}

	public int GetInputSize() => _weights.Length;
	public int GetUnits() => _weights[0].Length;

	public double[][] GetWeights() => _weights;
	public double[] GetBiases() => _biases;

	public void SetWeightsAndBiases(double[][] weights, double[] biases)
	{
		SetWeights(weights);
		SetBiases(biases);
	}

	public void SetWeights(double[][] weights)
	{
		if (weights.Length != GetInputSize())
			throw new ArgumentException("Weight size must be equal to input size. (number of features)");
		if (weights[0].Length != GetUnits())
			throw new ArgumentException("Single weight size must be equal to number of units.");
		_weights = weights;
	}

	public void SetBiases(double[] biases)
	{
		if (biases.Length != GetUnits())
			throw new ArgumentException("Biases size must be equal to number of units.");

		_biases = biases;
	}

public double[] Forward(double[] x) => Forward(x, new double[GetUnits()]);
    public double[] Forward(double[] x, double[] outputCache) => Activation.Forward(_activationType, x, _weights, _biases, outputCache);

    public (double[][] dw, double[] db, double[][] dx) Backward(double[] x, double[] o, LayerCache cache)
	{
		// Use pre-allocated cache arrays
		var (dw, db, dx) = Activation.Backward(_activationType, x, _weights, _biases, o, cache.Dw, cache.Db, cache.Dx);
		return (dw, db, dx);
	}

	public double[][] Forward(double[][] X) => X.Select(Forward).ToArray();

	public LayerCache CreateCache() => new LayerCache(GetInputSize(), GetUnits());
}
