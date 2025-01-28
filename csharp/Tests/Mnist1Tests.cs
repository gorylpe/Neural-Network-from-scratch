using NeuralNetworkFromScratch;
using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;
using NeuralNetworkFromScratch.Regularizers;
using NumSharp;

namespace Tests;

public class Mnist1Tests
{
	private double[][] X;
	private double[]   Y;

	[OneTimeSetUp]
	public void Setup()
	{
		var Xnp = np.load("data/mnist_1/X.npy");
		var rows = Xnp.shape[0];
		var cols = Xnp.shape[1];
		var Xflat = Xnp.ToArray<double>();

		X = new double[rows][];
		for (var i = 0; i < rows; i++)
		{
			X[i] = new double[cols];
			Array.Copy(Xflat, i * cols, X[i], 0, cols);
		}

		var Ynp = np.load("data/mnist_1/y.npy");
		Y = Ynp.ToArray<byte>().Select(Convert.ToDouble).ToArray();
	}

	[Test]
	public void TestLeakyReLU()
	{
		var model = new Model([
			new Dense(25, 400, ActivationType.LeakyReLU),
			new Dense(15, 25, ActivationType.LeakyReLU),
			new Dense(1, 15, ActivationType.Linear)
		], new Random(1234));

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 4, 0.01, 64);
		CheckResults(model);
	}

	[Test]
	public void TestReLUWithL2Regularization()
	{
		var model = new Model([
			new Dense(25, 400, ActivationType.ReLU, kernelRegularizer: new L2Regularizer(1)),
			new Dense(15, 25, ActivationType.ReLU, kernelRegularizer: new L2Regularizer(0.1)),
			new Dense(1, 15, ActivationType.Linear)
		], new Random(1234));

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 20, 0.005, 16);
		CheckResults(model);
	}

	private void CheckResults(Model model)
	{
		var Yhat = model.Predict(X).Select(y => y[0]).ToArray();
		Yhat = Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}
}
