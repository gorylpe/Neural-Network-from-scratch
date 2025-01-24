using System.Numerics;
using System.Text;
using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;

namespace NeuralNetworkFromScratch;

public class Model
{
	private readonly List<Dense> _layers;
	private List<LayerCache[]> _batchCaches;

	public Model(List<Dense> layers)
	{
		VerifyLayers(layers);
		_layers = layers;
	}

	private void VerifyLayers(List<Dense> layers)
	{
		int? previousLayerUnits = null;
		foreach (var layer in layers)
		{
			if (previousLayerUnits != null && layer.GetInputSize() != previousLayerUnits)
				throw new ArgumentException(
					$"The input size of the layer {layer.GetInputSize()} does not match the previous layer's output size {previousLayerUnits}.");

			previousLayerUnits = layer.GetUnits();
		}
	}

	public Dense GetLayer(int index) => _layers[index];

	public double[] Predict(double[] x) => _layers.Aggregate(x, (current, layer) => layer.Forward(current));
	public double[][] Predict(double[][] X) => _layers.Aggregate(X, (current, layer) => layer.Forward(current));

	public void InitializeCaches(int batchSize)
	{
		_batchCaches = new List<LayerCache[]>();
		foreach (var layer in _layers)
		{
			var layerCaches = new LayerCache[batchSize];
			for (int i = 0; i < batchSize; i++)
				layerCaches[i] = layer.CreateCache();
			_batchCaches.Add(layerCaches);
		}
	}

	private (double loss, double[][][] dwTotal, double[][] dbTotal) ComputeLossAndDerivatives(double[] x, double y, ILossCalc lossCalc, int exampleIndex)
	{
		var activations = ForwardPass(x, exampleIndex);
		var loss = lossCalc.Loss(y, activations[^1][0]);
		var dl = lossCalc.Derivative(y, activations[^1][0]);
		var (dwTotal, dbTotal) = BackwardPass(x, activations, dl, exampleIndex);
		return (loss, dwTotal, dbTotal);
	}

	private double[][] ForwardPass(double[] x, int exampleIndex)
	{
		var activations = new double[_layers.Count][];
		for (var i = 0; i < _layers.Count; i++)
		{
			var layer = _layers[i];
			var cache = _batchCaches[i][exampleIndex];
			activations[i] = layer.Forward(x, cache.ForwardOutput);
			x = activations[i];
		}
		return activations;
	}

	private (double[][][] dwTotal, double[][] dbTotal) BackwardPass(double[] x, double[][] activations, double dl, int exampleIndex)
	{
		var dwTotal = new double[_layers.Count][][];
		var dbTotal = new double[_layers.Count][];
		var dxSumPrevious = Enumerable.Repeat(dl, activations[^1].Length).ToArray();

		for (var layer = _layers.Count - 1; layer >= 0; layer--)
		{
			var input = layer == 0 ? x : activations[layer - 1];
			var cache = _batchCaches[layer][exampleIndex];
			var (dw, db, dx) = _layers[layer].Backward(input, activations[layer], cache);

			dxSumPrevious = ApplyDerivatives(dw, db, dx, dxSumPrevious);

			dwTotal[layer] = dw;
			dbTotal[layer] = db;
		}

		return (dwTotal, dbTotal);
	}

	private double[] ApplyDerivatives(double[][] dw, double[] db, double[][] dx, double[] dxSumPrevious)
	{
		var dxSum = new double[dx.Length];

		for (var i = 0; i < dw.Length; i++)
			for (var unit = 0; unit < dw[i].Length; unit++)
				dw[i][unit] *= dxSumPrevious[unit];

		for (var unit = 0; unit < db.Length; unit++)
			db[unit] *= dxSumPrevious[unit];

		for (var i = 0; i < dx.Length; i++)
		{
			dxSum[i] = 0;
			for (var unit = 0; unit < dx[i].Length; unit++)
				dxSum[i] += dx[i][unit] * dxSumPrevious[unit];
		}

		return dxSum;
	}

	private double[][][] AverageWeightsGradient(List<double[][][]> dwTotalExamples)
	{
		var avgGradients = new double[_layers.Count][][];

		for (var layer = 0; layer < _layers.Count; layer++)
		{
			var layerShape = _layers[layer].GetWeights();
			avgGradients[layer] = layerShape.Select(row => new double[row.Length]).ToArray();

			foreach (var example in dwTotalExamples)
			{
				for (var i = 0; i < layerShape.Length; i++)
					for (var unit = 0; unit < layerShape[i].Length; unit++)
						avgGradients[layer][i][unit] += example[layer][i][unit];
			}

			for (var i = 0; i < avgGradients[layer].Length; i++)
				for (var unit = 0; unit < avgGradients[layer][i].Length; unit++)
					avgGradients[layer][i][unit] /= dwTotalExamples.Count;
		}

		return avgGradients;
	}

	private double[][] AverageBiasesGradient(List<double[][]> dbTotalExamples)
	{
		var avgBiases = new double[_layers.Count][];

		for (var layer = 0; layer < _layers.Count; layer++)
		{
			var layerShape = _layers[layer].GetBiases();
			avgBiases[layer] = new double[layerShape.Length];

			foreach (var example in dbTotalExamples)
				for (var unit = 0; unit < layerShape.Length; unit++)
					avgBiases[layer][unit] += example[layer][unit];

			for (var unit = 0; unit < avgBiases[layer].Length; unit++)
				avgBiases[layer][unit] /= dbTotalExamples.Count;
		}

		return avgBiases;
	}

	private void UpdateWeightsAndBiasesByGradientDescent(double[][][] dwAvg, double[][] dbAvg, double learningRate)
	{
		for (var layer = 0; layer < _layers.Count; layer++)
		{
			var layerWeights = _layers[layer].GetWeights();
			var layerBiases = _layers[layer].GetBiases();

			for (var i = 0; i < layerWeights.Length; i++)
				for (var unit = 0; unit < layerWeights[i].Length; unit++)
					layerWeights[i][unit] -= learningRate * dwAvg[layer][i][unit];

			for (var unit = 0; unit < layerBiases.Length; unit++)
				layerBiases[unit] -= learningRate * dbAvg[layer][unit];
		}
	}

	private static bool ShouldLogProgress(int epoch, int totalEpochs, int count) => epoch % (totalEpochs / count) == 0 || epoch == totalEpochs - 1;

	public override string ToString()
	{
		var sb = new StringBuilder();
		for (var i = 0; i < _layers.Count; i++)
		{
			var layer = _layers[i];
			var w = layer.GetWeights();
			var b = layer.GetBiases();
			sb.AppendLine($"Layer {i}");
			for (var unit = 0; unit < layer.GetUnits(); unit++)
			{
				for (var j = 0; j < layer.GetInputSize(); j++) sb.Append($"{w[j][unit]:F2}, ");
				sb.AppendLine($"{b[unit]:F2}");
			}
		}

		return sb.ToString();
	}

	public void Fit(double[][] X, double[] Y, ILossCalc lossCalc, int epochs = 1000, double learningRate = 0.001, int batchSize = 16, int progressLogCount = 10)
	{
		foreach (var layer in _layers)
			layer.InitializeWeightsForTraining();

		InitializeCaches(batchSize);

		for (var epoch = 0; epoch < epochs; epoch++)
		{
			var lAvg = 0.0;
			for (var batchStart = 0; batchStart < X.Length; batchStart += batchSize)
			{
				var batchEnd = Math.Min(batchStart + batchSize, X.Length);
				var xBatch = X[batchStart..batchEnd];
				var yBatch = Y[batchStart..batchEnd];

				var gradients = ComputeGradients(xBatch, yBatch, lossCalc);
				UpdateWeightsAndBiasesByGradientDescent(gradients.dwAvg, gradients.dbAvg, learningRate);
				lAvg = gradients.lAvg;
			}

			if (ShouldLogProgress(epoch, epochs, progressLogCount))
			{
				Console.WriteLine($"Epoch {epoch + 1}");
				// Console.WriteLine(this.ToString());
				Console.WriteLine($"Average loss: {lAvg}");
			}
		}
	}

	private (double[][][] dwAvg, double[][] dbAvg, double lAvg) ComputeGradients(double[][] X, double[] Y, ILossCalc lossCalc)
	{
		var dwTotalExamples = new List<double[][][]>();
		var dbTotalExamples = new List<double[][]>();
		var lossTotalExamples = new List<double>();

		Parallel.For(0, X.Length, i =>
		{
			var (loss, dwTotal, dbTotal) = ComputeLossAndDerivatives(X[i], Y[i], lossCalc, i);
			lock (dwTotalExamples) dwTotalExamples.Add(dwTotal);
			lock (dbTotalExamples) dbTotalExamples.Add(dbTotal);
			lock (lossTotalExamples) lossTotalExamples.Add(loss);
		});

		var dwAvg = AverageWeightsGradient(dwTotalExamples);
		var dbAvg = AverageBiasesGradient(dbTotalExamples);
		var lAvg = lossTotalExamples.Average();

		return (dwAvg, dbAvg, lAvg);
	}
}
