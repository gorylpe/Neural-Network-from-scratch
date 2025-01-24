using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;
using NumSharp;

namespace NeuralNetworkFromScratch;

class Program
{
	static void Main(string[] args)
	{
		Mnist1();
	}

	private static void Mnist1()
	{
		var Xnp = np.load("data/mnist_1/X.npy");
		var rows = Xnp.shape[0];
		var cols = Xnp.shape[1];
		var Xflat = Xnp.ToArray<double>();

		var X = new double[rows][];
		for (var i = 0; i < rows; i++)
		{
			X[i] = new double[cols];
			Array.Copy(Xflat, i * cols, X[i], 0, cols);
		}

		var Ynp = np.load("data/mnist_1/y.npy");
		var Y = Ynp.ToArray<byte>().Select(Convert.ToDouble).ToArray();
		
		var model = new Model([
			new Dense(25, 400, ActivationType.ReLU),
			new Dense(15, 25, ActivationType.ReLU),
			new Dense(1, 15, ActivationType.Linear)
		]);

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 100, 0.03, 128);

		var Yhat = model.Predict(X).Select(y => y[0]).ToArray();
		Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
	}
}
