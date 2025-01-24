using System.Diagnostics.CodeAnalysis;

namespace NeuralNetworkFromScratch.Regularizers;

[Obsolete("Experimental could not work")]
public class L2Regularizer : IKernelRegularizer
{
	private readonly double _lambda;

	public L2Regularizer(double lambda)
	{
		if (lambda <= 0)
			throw new ArgumentException("lambda must be positive.");
		_lambda = lambda;
	}

	public double Loss(double[][] weights)
	{
		var regTerm = weights.SelectMany(t => t).Sum(t1 => t1 * t1);
		return 0.5 * _lambda * regTerm;
	}

	public void Regularize(double[][] dwRegularized, double[][] weights)
	{
		for (var i = 0; i < dwRegularized.Length; i++)
		for (var j = 0; j < dwRegularized[i].Length; j++)
			dwRegularized[i][j] = _lambda * weights[i][j];
	}
}
