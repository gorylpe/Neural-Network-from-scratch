using NeuralNetworkFromScratch.Layers;

namespace NeuralNetworkFromScratch;

class Program
{
	static void Main(string[] args)
	{
		var (X, Y) = TestData.LoadCoffeeData(10000);
		var norm = new Normalization(2);
		norm.Adapt(X);
		var xnorm = norm.Forward(X);

		var model = new Model([
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

		for (int i = 0; i < 10; i++)
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
}
