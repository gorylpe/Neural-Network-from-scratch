using NumSharp;

namespace NeuralNetworkFromScratch;

public static class Activation
{
	private const double SigmoidThreshold = 5; // Prevent gradient vanishing
	private const double LeakyReLUAlpha   = 0.01;

	public static double[] Forward(ActivationType activationType, double[] x, double[][] w, double[] b, double[] outputCache)
	{
		var linear = Linear(x, w, b, outputCache);
		return activationType switch
		{
			ActivationType.Linear    => linear,
			ActivationType.Sigmoid   => Sigmoid(linear),
			ActivationType.ReLU      => ReLU(linear),
			ActivationType.LeakyReLU => LeakyReLU(linear),
			ActivationType.Softmax   => throw new NotImplementedException(),
			_                        => throw new ArgumentOutOfRangeException(nameof(activationType), activationType, null)
		};
	}

	public static (double[][] dw, double[] db, double[][] dx) Backward(ActivationType activationType, double[] x,
		double[][] w, double[] b, double[] o, double[][] dwCache, double[] dbCache, double[][] dxCache)
	{
		var linear = LinearDerivative(x, w, b, dwCache, dbCache, dxCache);
		return activationType switch
		{
			ActivationType.Linear    => linear,
			ActivationType.Sigmoid   => SigmoidDerivative(linear, o),
			ActivationType.ReLU      => ReLUDerivative(linear, o),
			ActivationType.LeakyReLU => LeakyReLUDerivative(linear, o),
			ActivationType.Softmax   => throw new NotImplementedException(),
			_                        => throw new ArgumentOutOfRangeException(nameof(activationType), activationType, null)
		};
	}

	public static double[] Linear(double[] x, double[][] w, double[] b, double[] outputCache)
	{
		// todo use np.matmul(A_in, W) + B
		Array.Copy(b, outputCache, outputCache.Length);
		var oLen = w[0].Length;
		for (var i = 0; i < x.Length; i++)
		for (var j = 0; j < oLen; j++)
			outputCache[j] += x[i] * w[i][j];

		return outputCache;
	}

	public static double[] Sigmoid(double[] o)
	{
		for (var i = 0; i < o.Length; i++)
			o[i] = Sigmoid(o[i]);
		return o;
	}

	public static double Sigmoid(double o) => 1.0 / (1.0 + Math.Exp(Math.Clamp(-o, -SigmoidThreshold, SigmoidThreshold)));

	public static double[] ReLU(double[] o)
	{
		for (var i = 0; i < o.Length; i++)
			o[i] = ReLU(o[i]);
		return o;
	}

	private static double ReLU(double o) => Math.Max(0, o);


	private static double[] LeakyReLU(double[] o)
	{
		for (var i = 0; i < o.Length; i++)
			o[i] = LeakyReLU(o[i]);
		return o;
	}

	private static double LeakyReLU(double o) => o > 0 ? o : LeakyReLUAlpha * o;

	public static (double[][] dw, double[] db, double[][] dx) LinearDerivative(double[] x,       double[][] w,       double[]   b,
		double[][]                                                                      dwCache, double[]   dbCache, double[][] dxCache)
	{
		for (var i = 0; i < dwCache.Length; i++)
		{
			for (var unit = 0; unit < w[i].Length; unit++)
				dwCache[i][unit] = x[i];
		}

		for (var i = 0; i < b.Length; i++)
			dbCache[i] = 1;

		for (var i = 0; i < x.Length; i++)
		{
			for (var unit = 0; unit < w[i].Length; unit++)
				dxCache[i][unit] = w[i][unit];
		}

		return (dwCache, dbCache, dxCache);
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

	public static (double[][] dw, double[] db, double[][] dx) LeakyReLUDerivative((double[][] dw, double[] db, double[][] dx) linearDerivative, double[] o)
	{
		var (dw, db, dx) = linearDerivative;
		var rd = new double[o.Length];
		for (var i = 0; i < rd.Length; i++)
			rd[i] = o[i] > 0 ? 1 : LeakyReLUAlpha;

		ApplyDerivativeChainRule(rd, dw, db, dx);
		return (dw, db, dx);
	}
}
