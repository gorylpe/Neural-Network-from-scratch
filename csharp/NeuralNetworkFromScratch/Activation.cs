namespace NeuralNetworkFromScratch;

public static class Activation
{
	public static double[] Forward(ActivationType activationType, double[] x, double[][] w, double[] b) =>
		activationType switch
		{
			ActivationType.Linear  => Linear(x, w, b),
			ActivationType.Sigmoid => Sigmoid(Linear(x, w, b)),
			ActivationType.ReLU    => ReLU(Linear(x, w, b)),
			ActivationType.Softmax => throw new NotImplementedException(),
			_                      => throw new ArgumentOutOfRangeException(nameof(activationType), activationType, null)
		};

	public static (double[][] dw, double[] db, double[][] dx) Backward(ActivationType activationType, double[] x,
		double[][]                                                                    w,              double[] b, double[] o) =>
		activationType switch
		{
			ActivationType.Linear  => LinearDerivative(x, w, b),
			ActivationType.Sigmoid => SigmoidDerivative(LinearDerivative(x, w, b), o),
			ActivationType.ReLU    => ReLUDerivative(LinearDerivative(x, w, b), o),
			ActivationType.Softmax => throw new NotImplementedException(),
			_                      => throw new ArgumentOutOfRangeException(nameof(activationType), activationType, null)
		};

	public static double[] Linear(double[] x, double[][] w, double[] b)
	{
		// todo use np.matmul(A_in, W) + B
		var oLen = w[0].Length;
		var o = (double[])b.Clone();
		for (var i = 0; i < x.Length; i++)
		for (var j = 0; j < oLen; j++)
			o[j] += x[i] * w[i][j];

		return o;
	}

	public static double[] Sigmoid(double[] o)
	{
		for (var i = 0; i < o.Length; i++)
			o[i] = 1.0 / (1.0 + Math.Exp(-o[i]));
		return o;
	}

	public static double[] ReLU(double[] o)
	{
		for (var i = 0; i < o.Length; i++)
			o[i] = Math.Max(0, o[i]);
		return o;
	}

	public static (double[][] dw, double[] db, double[][] dx) LinearDerivative(double[] x, double[][] w, double[] b)
	{
		var dw = new double[w.Length][];
		var db = new double[b.Length];
		var dx = new double[x.Length][];

		for (var i = 0; i < dw.Length; i++)
		{
			dw[i] = new double[w[i].Length];
			for (var unit = 0; unit < w[i].Length; unit++)
				dw[i][unit] = x[i];
		}

		for (var i = 0; i < b.Length; i++)
			db[i] = 1;

		for (var i = 0; i < x.Length; i++)
		{
			dx[i] = new double[w[i].Length];
			for (var unit = 0; unit < w[i].Length; unit++)
				dx[i][unit] = w[i][unit];
		}

		return (dw, db, dx);
	}

	public static (double[][] dw, double[] db, double[][] dx) SigmoidDerivative((double[][] dw, double[] db, double[][] dx) linearDerivative, double[] o)
	{
		var (dw, db, dx) = linearDerivative;
		var sd = new double[o.Length];
		for (var i = 0; i < sd.Length; i++)
			sd[i] = o[i] * (1 - o[i]);

		ApplyDerivativeChainRule(sd, dw, db, dx);
		return (dw, db, dx);
	}

	private static void ApplyDerivativeChainRule(double[] dchain, double[][] dw, double[] db, double[][] dx)
	{
		for (var i = 0; i < dw.Length; i++)
		for (var unit = 0; unit < dw[i].Length; unit++)
			dw[i][unit] *= dchain[unit];
		for (var unit = 0; unit < db.Length; unit++)
			db[unit] *= dchain[unit];
		for (var i = 0; i < dx.Length; i++)
		for (var unit = 0; unit < dx[i].Length; unit++)
			dx[i][unit] *= dchain[unit];
	}

	public static (double[][] dw, double[] db, double[][] dx) ReLUDerivative((double[][] dw, double[] db, double[][] dx) linearDerivative, double[] o)
	{
		var (dw, db, dx) = linearDerivative;
		var rd = new double[o.Length];
		for (var i = 0; i < rd.Length; i++)
			rd[i] = o[i] > 0 ? 1 : 0;

		ApplyDerivativeChainRule(rd, dw, db, dx);
		return (dw, db, dx);
	}
}
