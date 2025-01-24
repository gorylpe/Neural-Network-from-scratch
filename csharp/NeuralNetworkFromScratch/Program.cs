using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;
using NeuralNetworkFromScratch.Regularizers;
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
		
		// var rnd = new Random(1234);
		// var zipped = X.Zip(Y).OrderBy(x => rnd.Next()).ToList();
		// X = zipped.Select(x => x.Item1).ToArray();
		// Y = zipped.Select(x => x.Item2).ToArray();
		
		// Works with leaky relu or relu and batch norm
		var model = new Model([
			new Dense(25, 400, ActivationType.LeakyReLU),
			new Dense(15, 25, ActivationType.LeakyReLU),
			new Dense(1, 15, ActivationType.Linear)
		], random: new Random(1234));

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 100, 0.01, 128);
		
		var Yhat = model.Predict(X).Select(y => y[0]).ToArray();
		Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
	}
}
