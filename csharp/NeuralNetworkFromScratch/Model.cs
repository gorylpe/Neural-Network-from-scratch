using System.Numerics;
using System.Text;
using NeuralNetworkFromScratch.Layers;
using NeuralNetworkFromScratch.Loss;

namespace NeuralNetworkFromScratch;

public class Model
{
	private readonly List<Dense> _layers;

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

	public double[]   Predict(double[] x)   => _layers.Aggregate(x, (current, layer) => layer.Forward(current));
	public double[][] Predict(double[][] X) => _layers.Aggregate(X, (current, layer) => layer.Forward(current));

	public void Fit(double[][] X, double[] Y, int epochs = 1000, double learningRate = 0.001)
	{
		// var loss = new BinaryCrossEntropy();
		var loss = new MeanSquaredError();

		for (var epoch = 0; epoch < epochs; epoch++)
		{
			var dwTotalExamples = new List<double[][][]>();
			var dbTotalExamples = new List<double[][]>();
			var lTotalExamples = new List<double>();

			foreach (var (x, y) in X.Zip(Y))
			{
				var o = new double[_layers.Count][];
				var dwTotal = new double[_layers.Count][][];
				var dbTotal = new double[_layers.Count][];
				var dxsTotal = new double[_layers.Count][];

				for (var i = 0; i < _layers.Count; i++)
					o[i] = _layers[i].Forward(x);

				var l = loss.Loss(y, o[^1][0]);
				var dl = loss.Derivative(y, o[^1][0]);

				for (var layer = _layers.Count - 1; layer >= 0; layer--)
				{
					var input = layer == 0 ? x : o[layer - 1];
					var (dw, db, dx) = _layers[layer].Backward(input, o[layer]);
					var dxs = new double[dx.Length];

					if (layer == _layers.Count - 1)
					{
						for (var i = 0; i < dw.Length; i++)
						for (var unit = 0; unit < dw[i].Length; unit++)
							dw[i][unit] *= dl;
						for (var unit = 0; unit < db.Length; unit++)
							db[unit] *= dl;

						for (var i = 0; i < dx.Length; i++)
						{
							dxs[i] = 0;
							for (var unit = 0; unit < dx[i].Length; unit++)
								dxs[i] += dx[i][unit] * dl;
						}
					}
					else
					{
						var prevDxs = dxsTotal[layer + 1];

						for (var i = 0; i < dw.Length; i++)
						for (var unit = 0; unit < dw[i].Length; unit++)
							dw[i][unit] *= prevDxs[unit];
						for (var unit = 0; unit < db.Length; unit++)
							db[unit] *= prevDxs[unit];

						for (var i = 0; i < dx.Length; i++)
						{
							dxs[i] = 0;
							for (var unit = 0; unit < dx[i].Length; unit++)
								dxs[i] += dx[i][unit] * prevDxs[unit];
						}
					}

					dwTotal[layer] = dw;
					dbTotal[layer] = db;
					dxsTotal[layer] = dxs;
				}

				dwTotalExamples.Add(dwTotal);
				dbTotalExamples.Add(dbTotal);
				lTotalExamples.Add(l);
			}

			var dwAvg = new double[dwTotalExamples[0].Length][][];
			var dbAvg = new double[dbTotalExamples[0].Length][];
			var lAvg = lTotalExamples.Average();
			
			for (var layer = 0; layer < _layers.Count; layer++)
			{
				var layerWeights = _layers[layer].GetWeights();
				var layerBiases = _layers[layer].GetBiases();

				dwAvg[layer] = new double[layerWeights.Length][];
				for (var i = 0; i < layerWeights.Length; i++)
				{
					dwAvg[layer][i] = new double[layerWeights[i].Length];
					for (var unit = 0; unit < layerWeights[i].Length; unit++)
					{
						foreach (var example in dwTotalExamples)
							dwAvg[layer][i][unit] += example[layer][i][unit];
						dwAvg[layer][i][unit] /= dwTotalExamples.Count;
					}
				}

				dbAvg[layer] = new double[layerBiases.Length];
				for (var unit = 0; unit < layerBiases.Length; unit++)
				{
					foreach (var example in dbTotalExamples)
						dbAvg[layer][unit] += example[layer][unit];
					dbAvg[layer][unit] /= dbTotalExamples.Count;
				}
			}

			// Gradient descent
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
			
			
			if (epoch % 100 == 0 || epoch == epochs - 1)
			{
				Console.WriteLine($"Epoch {epoch + 1}\n{this}");
				Console.WriteLine($"Average loss: {lAvg}");
			}
		}
	}

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
				sb.AppendLine($"{   b[unit]:F2}");
			}
		}

		return sb.ToString();
	}
}
