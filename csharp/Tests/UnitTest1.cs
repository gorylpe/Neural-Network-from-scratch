using NeuralNetworkFromScratch;
using NeuralNetworkFromScratch.Layers;

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
}
