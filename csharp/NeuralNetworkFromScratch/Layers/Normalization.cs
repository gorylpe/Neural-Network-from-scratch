using System.Numerics;

namespace NeuralNetworkFromScratch.Layers;

public class Normalization(int units, double[]? mean = default, double[]? std = default)
	: ILayer
{
	private double[] _mean = mean ?? new double[units];
	private double[] _std  = std ?? new double[units];

	public void Adapt(double[][] x)
	{
		var mean = new double[units];
		foreach (var v in x)
		{
			for (var i = 0; i < units; i++)
				mean[i] += v[i];
		}

		for (var i = 0; i < units; i++)
			mean[i] /= x.Length;

		var std = new double[units];
		foreach (var v in x)
		{
			for (var i = 0; i < units; i++)
			{
				var tmp = v[i] - mean[i];
				std[i] += tmp * tmp;
			}
		}

		for (var i = 0; i < units; i++)
		{
			std[i] /= x.Length;
			std[i] = Math.Sqrt(std[i]);
		}

		_mean = mean;
		_std = std;
	}

	public (double RegularizationLoss, double[] Activations) Forward(double[] x)
	{
		var o = new double[x.Length];
		for (var i = 0; i < x.Length; i++)
			o[i] = (x[i] - _mean[i]) / _std[i];
		return (0.0, o);
	}

	public void InitializeWeightsForTraining(Random? random = null)
	{
	}
}
