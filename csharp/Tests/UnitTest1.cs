using NeuralNetworkFromScratch;
using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;
using NumSharp;

namespace Tests;

public class Tests
{
	[TestCase(400, 2)]
	[TestCase(2000, 2)]
	[TestCase(1000, 3)]
	[TestCase(1000, 69)]
	public void TestCoffeeDataWithPrecomputedWeights(int examples, int seed)
	{
		var (X, Y) = TestData.LoadCoffeeData(examples, seed);
		var norm = new Normalization(2);
		norm.Adapt(X);
		var xnorm = norm.Forward(X);

		var model = new Model([
			new Dense(3, 2, ActivationType.Sigmoid),
			new Dense(1, 3, ActivationType.Sigmoid)
		]);

		model.GetLayer(0).SetWeightsAndBiases([
			[-8.94, 0.29, 12.89],
			[-0.17, -7.34, 10.79]
		], [-9.87, -9.28, 1.01]);

		model.GetLayer(1).SetWeightsAndBiases([
				[-31.38],
				[-27.86],
				[-32.79]
			], [15.54]
		);

		var Yhat = model.Predict(xnorm);
		var errors = Yhat.Select((yhat, i) => (yhat[0] > 0.5 ? 1 : 0) == Y[i]).Count(x => !x);
		var accuracy = 1.0 - 1.0 * errors / Y.Length;
		Console.WriteLine($"Accuracy: {accuracy:P}, {errors} errors out of {Y.Length} examples");
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}

	[Test]
	public void TestModelForLinearRegression()
	{
		var (X, Y) = TestData.LoadLinearData();
		var model = new Model([
			new Dense(1, 1, ActivationType.Linear)
		]);
		model.Fit(X, Y, new MeanSquaredError(), 2000, 0.01);
		var Yhat = model.Predict(X);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		for (var i = 0; i < Y.Length; i++)
			Assert.That(Yhat[i][0] / Y[i], Is.LessThan(1.01).And.GreaterThan(0.99));
	}

	[Test]
	public void TestModelForLogisticRegression()
	{
		var (X, Y) = TestData.LoadLogisticData();
		var model = new Model([
			new Dense(1, 1, ActivationType.Sigmoid),
		]);

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: false), 5000, 0.005);
		var Yhat = model.Predict(X);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}

	[Test]
	public void TestModelForLogisticRegressionFromLogits()
	{
		var (X, Y) = TestData.LoadLogisticData();
		var model = new Model([
			new Dense(1, 1, ActivationType.Linear),
		]);

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 5000, 0.005);
		var Yhat = model.Predict(X).Select(y => y[0]).ToArray();
		Yhat = Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}

	[Test]
	public void TestModelForCoffeeDataUsingLogits()
	{
		var (X, Y) = TestData.LoadCoffeeData(400);
		var norm = new Normalization(2);
		norm.Adapt(X);
		var Xnorm = norm.Forward(X);

		var model = new Model([
			new Dense(3, 2, ActivationType.Sigmoid),
			new Dense(1, 3, ActivationType.Linear)
		]);

		model.Fit(Xnorm, Y, new BinaryCrossEntropy(fromLogits: true), 5000, 0.5);

		var Yhat = model.Predict(Xnorm).Select(y => y[0]).ToArray();
		Yhat = Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Y, Yhat);
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}

	[Test]
	public void TestModelWithReLuActivation()
	{
		// Simple linear data
		var X = new double[][] { [1], [2], [3], [4] };
		var Y = new double[] { 2, 4, 6, 8 };

		// Model with ReLU activation
		var model = new Model([
			new Dense(10, 1, ActivationType.ReLU), // Hidden layer with ReLU
			new Dense(1, 10, ActivationType.Linear) // Output layer
		]);

		// Train the model
		model.Fit(X, Y, new MeanSquaredError(), epochs: 1000, learningRate: 0.01);

		// Test predictions
		var Yhat = model.Predict(X);
		for (var i = 0; i < Y.Length; i++)
		{
			Assert.That(Yhat[i][0], Is.EqualTo(Y[i]).Within(0.1));
		}
	}

	[Test]
	public void TestModelForMnist1()
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

		model.Fit(X, Y, new BinaryCrossEntropy(fromLogits: true), 100, 0.01, 64);

		var Yhat = model.Predict(X).Select(y => y[0]).ToArray();
		Yhat = Activation.Sigmoid(Yhat);
		Utils.ConsoleWriteYAndYHat(Y, Yhat);
		var accuracy = Utils.CalculateAndConsoleWriteAccuracy(Yhat, Y);
		Assert.That(accuracy, Is.GreaterThan(0.95));
	}
}
