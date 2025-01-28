using NeuralNetworkFromScratch;
using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;
using NeuralNetworkFromScratch.Regularizers;

namespace Tests
{
    public class L2RegularizationTests
    {
        [Test]
        public void TestL2RegularizationEffect()
        {
            // Arrange
            var inputSize = 2;
            var units = 1;
            var lambda = 0.1;
            
            // Create two identical models - one with L2, one without
            var modelNoReg = new Model([new Dense(units, inputSize, ActivationType.Linear)]);
            
            var modelWithReg = new Model([new Dense(units, inputSize, ActivationType.Linear, kernelRegularizer: new L2Regularizer(lambda))]);

            // Initialize with same weights
            var weights = new double[][] {[0.5],[-0.5]};
            var biases = new[] { 0.1 };
            modelNoReg.GetLayer(0).SetWeightsAndBiases(weights, biases);
            modelWithReg.GetLayer(0).SetWeightsAndBiases(weights, biases);

            // Simple linear data
            var X = new double[][] { [1.0, 2.0] };
            var Y = new[] { 3.0 };
            var lossCalc = new MeanSquaredError();
            Console.WriteLine(modelNoReg.ToString());
            Console.WriteLine(modelWithReg.ToString());

            // Act - Run one training step
            modelNoReg.Fit(X, Y, lossCalc, epochs: 1, learningRate: 0.1, initailizeWeights: false);
            modelWithReg.Fit(X, Y, lossCalc, epochs: 1, learningRate: 0.1, initailizeWeights: false);
            
            Console.WriteLine(modelNoReg.ToString());
            Console.WriteLine(modelWithReg.ToString());

            // Assert
            var weightsNoReg = modelNoReg.GetLayer(0).GetWeights().Select(x => x[0]).ToArray();
            var weightsWithReg = modelWithReg.GetLayer(0).GetWeights().Select(x => x[0]).ToArray();
            Assert.Multiple(() =>
            {
                // With L2 regularization, weights should be smaller
                Assert.That(weightsWithReg[0], Is.LessThan(weightsNoReg[0]));
                // Initial weight was negative so after regularization should be bigger
                Assert.That(weightsWithReg[1], Is.GreaterThan(weightsNoReg[1]));
            });

            // Check that regularization is actually being applied
            var expectedWeight1 = weightsNoReg[0] - lambda * weights[0][0] * 0.1;
            var expectedWeight2 = weightsNoReg[1] - lambda * weights[1][0] * 0.1;
            Assert.Multiple(() =>
            {
                Assert.That(expectedWeight1, Is.EqualTo(weightsWithReg[0]).Within(1e-6));
                Assert.That(expectedWeight2, Is.EqualTo(weightsWithReg[1]).Within(1e-6));
            });
        }
    }
}
