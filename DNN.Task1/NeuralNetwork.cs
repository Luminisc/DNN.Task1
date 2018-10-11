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
        public float[] Image;
        public byte Label;
    }

    class NeuralNetwork
    {
        public int InputLayerSize { get; protected set; }
        public int HiddenLayerSize { get; protected set; }
        public int OutputLayerSize { get; protected set; }
        public float LearningRate { get; protected set; }

        protected float[] InputLayer { get; set; }
        protected float[] HiddenLayer { get; set; }
        protected float[] OutputLayer { get; set; }

        protected float[] HiddenWeightsDeltas { get; set; }
        protected float[] OutputWeightsDeltas { get; set; }

        protected float[] HiddenLayerGradient { get; set; }
        protected float[] OutputLayerGradient { get; set; }

        protected float[,] HiddenWeights { get; set; }
        protected float[,] OutputWeights { get; set; }

        public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, float learningRate)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;
            LearningRate = learningRate;

            InputLayer = new float[inputLayerSize];
            HiddenLayer = new float[hiddenLayerSize];
            OutputLayer = new float[outputLayerSize];

            Random rnd = new Random();

            HiddenWeightsDeltas = new float[hiddenLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
            {
                HiddenWeightsDeltas[i] = (float)((rnd.NextDouble() - 0.5) * 2.0);
            }
            OutputWeightsDeltas = new float[outputLayerSize];
            for (int i = 0; i < outputLayerSize; i++)
            {
                OutputWeightsDeltas[i] = (float)((rnd.NextDouble() - 0.5) * 2.0);
            }

            HiddenLayerGradient = new float[hiddenLayerSize];
            Array.Clear(HiddenLayerGradient, 0, hiddenLayerSize);
            OutputLayerGradient = new float[outputLayerSize];
            Array.Clear(OutputLayerGradient, 0, outputLayerSize);

            HiddenWeights = new float[inputLayerSize, hiddenLayerSize];
            for (int i = 0; i < inputLayerSize; i++)
                for (int j = 0; j < hiddenLayerSize; j++)
                    HiddenWeights[i, j] = (float)((rnd.NextDouble() - 0.5) * 2.0);
            OutputWeights = new float[hiddenLayerSize, outputLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
                for (int j = 0; j < outputLayerSize; j++)
                    OutputWeights[i, j] = (float)((rnd.NextDouble() - 0.5) * 2.0);
        }

        public void Train(List<ImageDescription> combinedData, int epochsCount, float crossEntropyError)
        {
            var expectedOutput = new float[OutputLayerSize];
            
            Random rnd = new Random();
            Stopwatch sw = new Stopwatch();

            for (int i = 0; i < epochsCount; i++)
            {
                sw.Restart();
                int correctAnswers = 0;
                var randomizedData = combinedData.OrderBy(x => rnd.Next());

                foreach (var data in randomizedData)
                {
                    Array.Copy(data.Image, InputLayer, InputLayerSize);
                    Array.Clear(expectedOutput, 0, OutputLayerSize);
                    expectedOutput[data.Label] = 1.0f;

                    CalculateHiddenLayer();
                    CalculateOutputLayer();

                    if (expectedOutput[IndexOfMaximum()] > 0.0f)
                        correctAnswers++;

                    CalculateGradient(expectedOutput);

                    ChangeWeights();
                    ChangeWeightsDeltas();
                }

                var crossEnthropy = CalculateCrossEntropy(randomizedData.ToList());
                Console.WriteLine($"Cross enthropy of dataset: {crossEnthropy}");

                var accuracy = correctAnswers / combinedData.Count;
                Console.WriteLine($"Accuracy: {accuracy}");
                sw.Stop();
                Console.WriteLine($"Epoch finished in {sw.ElapsedMilliseconds} ms.");

                if ((crossEnthropy <= crossEntropyError) || (1 - accuracy <= crossEntropyError))
                    break;
            }
        }

        protected void CalculateHiddenLayer()
        {
            float sum = 0.0f;
            for (int i = 0; i < HiddenLayerSize; i++)
            {
                sum = 0.0f;
                for (int j = 0; j < InputLayerSize; j++)
                    sum += InputLayer[j] * HiddenWeights[j, i];
                sum += HiddenWeightsDeltas[i];
                HiddenLayer[i] = sum;
            }
        }

        protected void CalculateOutputLayer()
        {
            float sum = 0.0f;
            for (int i = 0; i < OutputLayerSize; i++)
            {
                sum = 0.0f;
                for (int j = 0; j < HiddenLayerSize; j++)
                    sum += HiddenLayer[j] * OutputWeights[j, i];
                sum += OutputWeightsDeltas[i];
                OutputLayer[i] = sum;
            }
            CalculateSoftmax();
        }

        protected void CalculateSoftmax()
        {
            float sum = 0;
            for (int i = 0; i < OutputLayerSize; i++)
            {
                sum += MathF.Exp(OutputLayer[i]);
            }
            for (int i = 0; i < OutputLayerSize; i++)
            {
                OutputLayer[i] /= sum;
            }
        }

        public void CalculateGradient(float[] expectedOutput)
        {
            for (int i = 0; i < OutputLayerSize; i++)
                OutputLayerGradient[i] = expectedOutput[i] - OutputLayer[i];

            float sum = 0.0f;
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

        public float CalculateCrossEntropy(List<ImageDescription> dataCollection)
        {
            float sum = 0;
            float[] x = new float[InputLayerSize];
            float[] y = new float[OutputLayerSize];
            float[] z = new float[OutputLayerSize];
            int size = dataCollection.Count;

            foreach (var data in dataCollection)
            {
                for (int j = 0; j < InputLayerSize; j++)
                    x[j] = data.Image[j];

                Array.Clear(z, 0, OutputLayerSize);
                z[data.Label] = 1.0f;

                InputLayer = x;
                CalculateHiddenLayer();
                CalculateOutputLayer();
                y = OutputLayer;
                for (int i = 0; i < OutputLayerSize; i++)
                    sum += MathF.Log(y[i] * z[i]);
            }

            return -sum / size;
        }

        public int IndexOfMaximum()
        {
            int index = 0;
            float val = float.MinValue;
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
