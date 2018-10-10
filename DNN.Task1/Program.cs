using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using DNN.Task1.DataSetContainers;

namespace DNN.Task1
{
    class Program
    {
        static string datasetPath = @"..\..\..\DataSet\";
        static string trainImagesPath = Path.Combine(datasetPath, "train-images.idx3-ubyte");
        static string trainLabelsPath = Path.Combine(datasetPath, "train-labels.idx1-ubyte");
        static string testImagesPath = Path.Combine(datasetPath, "t10k-images.idx3-ubyte");
        static string testLabelsPath = Path.Combine(datasetPath, "t10k-labels.idx1-ubyte");

        static void Main(string[] args)
        {
            var config = new
            {
                EpochsCount = 10,
                HiddenLayerSize = 10,
                CrossError = 0.005,
                LearningRate = 0.01f
            };

            Stopwatch sw = new Stopwatch();
            sw.Restart();
            ImagesContainer trainIC = new ImagesContainer(trainImagesPath);
            LabelsContainer trainLC = new LabelsContainer(trainLabelsPath);
            var imageSize = trainIC.ImagesWidth * trainIC.ImagesHeight;

            NeuralNetwork NN = new NeuralNetwork(imageSize, config.HiddenLayerSize, 10, config.LearningRate);




            sw.Stop();
            Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds} milliseconds");


            Console.ReadLine();
        }
    }
}
