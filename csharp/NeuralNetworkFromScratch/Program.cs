using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;

namespace NeuralNetworkFromScratch;

class Program
{
	static void Main(string[] args)
	{
		// ModelForLogisticRegression();
		ModelForCoffeeData();
	}

	private static void ModelForLogisticRegression()
	{
		var (X, Y) = TestData.LoadLogisticData();
		var model = new Model([
			new Dense(1, 1, ActivationType.Sigmoid),
			new Dense(1, 1, ActivationType.Sigmoid),
		]);
		model.Fit(X, Y, new BinaryCrossEntropy(), 5000, 0.5);
		var Yhat = model.Predict(X);
		Console.WriteLine(string.Join("; ", Y.Zip(Yhat.Select(x => x[0])).Select(x => $"{x.Item1} -> {x.Item2:F2}")));
		var errors = Yhat.Select((yhat, i) => (yhat[0] > 0.5 ? 1 : 0) == Y[i]).Count(x => !x);
		var accuracy = 1.0 - 1.0 * errors / Y.Length;
		Console.WriteLine($"Accuracy: {accuracy:P}, {errors} errors out of {Y.Length} examples");
	}

	private static void ModelForCoffeeData()
	{
		var (X, Y) = TestData.LoadCoffeeData(400);
		var norm = new Normalization(2);
		norm.Adapt(X);
		var xnorm = norm.Forward(X);

		// var model = LoadPrecomputed();
		var model = TrainModel(xnorm, Y);

		for (int i = 0; i < 1; i++)
		{
			var xreal = X[i];
			var x = xnorm[i];
			var y = Y[i];
			var yhat = model.Predict(x)[0] > 0.5 ? 1 : 0;
			Console.WriteLine($"{(y != yhat ? "ERROR " : "")}y = {y}, yhat = {yhat}, x = {xreal.PrettyString()}");
		}

		var Yhat = model.Predict(xnorm);
		var errors = Yhat.Select((yhat, i) => (yhat[0] > 0.5 ? 1 : 0) == Y[i]).Count(x => !x);
		var accuracy = 1.0 - 1.0 * errors / Y.Length;
		Console.WriteLine($"Accuracy: {accuracy:P}, {errors} errors out of {Y.Length} examples");
	}

	private static Model TrainModel(double[][] X, double[] Y)
	{
		var model = new Model([
			new Dense(3, 2, ActivationType.Sigmoid),
			new Dense(1, 3, ActivationType.Sigmoid)
		]);

		model.Fit(X, Y, new MeanSquaredError(), 10000, 0.1);
		return model;
	}

	private static Model LoadPrecomputed()
	{
		return new Model(
		[
			new Dense(3, 2, ActivationType.Sigmoid, [
					[-8.94, 0.29, 12.89],
					[-0.17, -7.34, 10.79]
				],
				[-9.87, -9.28, 1.01]),
			new Dense(1, 3, ActivationType.Sigmoid, [
					[-31.38],
					[-27.86],
					[-32.79]
				],
				[15.54])
		]);
	}
}
