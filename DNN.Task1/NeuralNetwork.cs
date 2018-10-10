using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.Task1
{
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
            OutputLayerGradient = new float[hiddenLayerSize];
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
    }
}
