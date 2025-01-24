namespace NeuralNetworkFromScratch.Layers;

public class LayerCache
{
	public double[]   ForwardOutput { get; }
	public double[][] Dw            { get; }
	public double[]   Db            { get; }
	public double[][] Dx            { get; }

	public LayerCache(int inputSize, int units)
	{
		ForwardOutput = new double[units];
		Dw = new double[inputSize][];
		for (var i = 0; i < inputSize; i++)
			Dw[i] = new double[units];
		Db = new double[units];
		Dx = new double[inputSize][];
		for (var i = 0; i < inputSize; i++)
			Dx[i] = new double[units];
	}
}
