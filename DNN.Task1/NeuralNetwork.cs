using System;
using System.Collections.Generic;
using System.Text;
using DNN.Task1.DataSetContainers;
using System.Linq;
using System.Diagnostics;

namespace DNN.Task1
{
    struct ImageDescription
    {
        public double[] Image;
        public byte Label;
    }

    class NeuralNetwork
    {
        public int InputLayerSize { get; protected set; }
        public int HiddenLayerSize { get; protected set; }
        public int OutputLayerSize { get; protected set; }
        public double LearningRate { get; protected set; }

        protected double[] InputLayer { get; set; }
        protected double[] HiddenLayer { get; set; }
        protected double[] OutputLayer { get; set; }

        protected double[] HiddenWeightsDeltas { get; set; }
        protected double[] OutputWeightsDeltas { get; set; }

        protected double[] HiddenLayerGradient { get; set; }
        protected double[] OutputLayerGradient { get; set; }

        protected double[,] HiddenWeights { get; set; }
        protected double[,] OutputWeights { get; set; }

        public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, double learningRate)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;
            LearningRate = learningRate;

            InputLayer = new double[inputLayerSize];
            HiddenLayer = new double[hiddenLayerSize];
            OutputLayer = new double[outputLayerSize];

            Random rnd = new Random();

            HiddenWeightsDeltas = new double[hiddenLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
            {
                HiddenWeightsDeltas[i] = (double)((rnd.NextDouble() - 0.5) * 2.0);
            }
            OutputWeightsDeltas = new double[outputLayerSize];
            for (int i = 0; i < outputLayerSize; i++)
            {
                OutputWeightsDeltas[i] = (double)((rnd.NextDouble() - 0.5) * 2.0);
            }

            HiddenLayerGradient = new double[hiddenLayerSize];
            Array.Clear(HiddenLayerGradient, 0, hiddenLayerSize);
            OutputLayerGradient = new double[outputLayerSize];
            Array.Clear(OutputLayerGradient, 0, outputLayerSize);

            HiddenWeights = new double[inputLayerSize, hiddenLayerSize];
            for (int i = 0; i < inputLayerSize; i++)
                for (int j = 0; j < hiddenLayerSize; j++)
                    HiddenWeights[i, j] = (double)((rnd.NextDouble() - 0.5) * 2.0);
            OutputWeights = new double[hiddenLayerSize, outputLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
                for (int j = 0; j < outputLayerSize; j++)
                    OutputWeights[i, j] = (double)((rnd.NextDouble() - 0.5) * 2.0);
        }

        public void Train(List<ImageDescription> combinedData, int epochsCount, double crossEntropyError)
        {
            var expectedOutput = new double[OutputLayerSize];

            Random rnd = new Random();
            Stopwatch sw = new Stopwatch();

            for (int i = 0; i < epochsCount; i++)
            {
                Console.WriteLine($"[Epoch #{i + 1}]");
                sw.Restart();
                int correctAnswers = 0;
                var randomizedData = combinedData.OrderBy(x => rnd.Next()).ToList();

                foreach (var data in randomizedData)
                {
                    Array.Copy(data.Image, InputLayer, InputLayerSize);
                    Array.Clear(expectedOutput, 0, OutputLayerSize);
                    expectedOutput[data.Label] = 1.0;

                    CalculateHiddenLayer();
                    CalculateOutputLayer();

                    if (expectedOutput[IndexOfMaximum()] > 0.0)
                        correctAnswers++;

                    CalculateGradient(expectedOutput);

                    ChangeWeights();
                    ChangeWeightsDeltas();
                }

                var crossEnthropy = CalculateCrossEntropy(randomizedData);
                Console.WriteLine($"Cross enthropy of dataset: {crossEnthropy}");

                double accuracy = correctAnswers / (double)combinedData.Count;
                Console.WriteLine($"Accuracy: {accuracy}");
                sw.Stop();
                Console.WriteLine($"Epoch finished in {sw.ElapsedMilliseconds} ms.");
                Console.WriteLine();
                if ((crossEnthropy <= crossEntropyError) || (1 - accuracy <= crossEntropyError))
                    break;
            }
        }

        protected void CalculateHiddenLayer()
        {
            double sum = 0.0;
            for (int i = 0; i < HiddenLayerSize; i++)
            {
                sum = 0.0;
                for (int j = 0; j < InputLayerSize; j++)
                    sum += InputLayer[j] * HiddenWeights[j, i];
                sum += HiddenWeightsDeltas[i];
                HiddenLayer[i] = 1.0 / (1.0 + Math.Exp(-sum));
            }
        }

        protected void CalculateOutputLayer()
        {
            double sum = 0.0;
            for (int i = 0; i < OutputLayerSize; i++)
            {
                sum = 0.0;
                for (int j = 0; j < HiddenLayerSize; j++)
                    sum += HiddenLayer[j] * OutputWeights[j, i];
                sum += OutputWeightsDeltas[i];
                OutputLayer[i] = sum;
            }
            CalculateSoftmax();
        }

        protected void CalculateSoftmax()
        {
            double sum = 0;
            double[] exponentialLayer = new double[OutputLayerSize];
            for (int i = 0; i < OutputLayerSize; i++)
            {
                exponentialLayer[i] = Math.Exp(OutputLayer[i]);
                sum += exponentialLayer[i];
            }
            for (int i = 0; i < OutputLayerSize; i++)
                OutputLayer[i] = exponentialLayer[i] / sum;
        }

        public void CalculateGradient(double[] expectedOutput)
        {
            for (int i = 0; i < OutputLayerSize; i++)
                OutputLayerGradient[i] = expectedOutput[i] - OutputLayer[i];

            double sum = 0.0;
            for (int i = 0; i < HiddenLayerSize; i++)
            {
                for (int j = 0; j < OutputLayerSize; j++)
                    sum += OutputLayerGradient[j] * OutputWeights[i, j];
                HiddenLayerGradient[i] = sum * (HiddenLayer[i] * (1 - HiddenLayer[i]));
            }
        }

        public void ChangeWeights()
        {
            for (int i = 0; i < HiddenLayerSize; i++)
                for (int j = 0; j < OutputLayerSize; j++)
                    OutputWeights[i, j] += LearningRate * OutputLayerGradient[j] * HiddenLayer[i];
            for (int i = 0; i < InputLayerSize; i++)
                for (int j = 0; j < HiddenLayerSize; j++)
                    HiddenWeights[i, j] += LearningRate * HiddenLayerGradient[j] * InputLayer[i];
        }

        public void ChangeWeightsDeltas()
        {
            for (int i = 0; i < OutputLayerSize; i++)
                OutputWeightsDeltas[i] += LearningRate * OutputLayerGradient[i];
            for (int i = 0; i < HiddenLayerSize; i++)
                HiddenWeightsDeltas[i] += LearningRate * HiddenLayerGradient[i];
        }

        public double CalculateCrossEntropy(List<ImageDescription> dataCollection)
        {
            double sum = 0;
            double[] expectedOutput = new double[OutputLayerSize];

            foreach (var data in dataCollection)
            {
                Array.Copy(data.Image, InputLayer, InputLayerSize);

                Array.Clear(expectedOutput, 0, OutputLayerSize);
                expectedOutput[data.Label] = 1.0;

                CalculateHiddenLayer();
                CalculateOutputLayer();
                for (int i = 0; i < OutputLayerSize; i++)
                    sum += Math.Log(OutputLayer[i]) * expectedOutput[i];
            }

            return -sum / dataCollection.Count;
        }

        public int IndexOfMaximum()
        {
            int index = 0;
            double val = double.MinValue;
            for (int i = 0; i < OutputLayerSize; i++)
                if (OutputLayer[i] > val)
                {
                    val = OutputLayer[i];
                    index = i;
                }
            return index;
        }
    }
}
