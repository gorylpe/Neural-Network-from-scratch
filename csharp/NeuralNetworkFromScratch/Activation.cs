using System.Numerics;

namespace NeuralNetworkFromScratch;

public static class Activation
{
	public static double[] Forward(ActivationType activationType, double[] x, double[][] w, double[] b)
	{
		return activationType switch
		{
			ActivationType.Linear => Linear(x, w, b),
			ActivationType.Sigmoid => Sigmoid(x, w, b),
			ActivationType.ReLU => ReLU(x, w, b),
			ActivationType.Softmax => throw new NotImplementedException(),
			_ => throw new ArgumentOutOfRangeException(nameof(activationType), activationType, null)
		};
	}

	public static double[] Linear(double[] x, double[][] w, double[] b)
	{
		var oLen = w[0].Length;
		var o = (double[])b.Clone();
		for (var i = 0; i < x.Length; i++)
		for (var j = 0; j < oLen; j++)
			o[j] += x[i] * w[i][j];

		return o;
	}

	public static double[] Sigmoid(double[] x, double[][] w, double[] b)
	{
		var o = Linear(x, w, b);
		for (var i = 0; i < o.Length; i++)
			o[i] = 1.0 / (1.0 + Math.Exp(-o[i]));
		return o;
	}

	public static double[] ReLU(double[] x, double[][] w, double[] b)
	{
		var o = Linear(x, w, b);
		for (var i = 0; i < o.Length; i++)
			o[i] = Math.Max(0, o[i]);
		return o;
	}
}
