using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using DNN.Task1.DataSetContainers;
using System.Collections.Generic;
using System.Reflection;

namespace DNN.Task1
{
    class Program
    {
        public static readonly string datasetPath = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), @"../../../DataSet/");
        //static string datasetPath = @"..\..\..\DataSet\";
        static string trainImagesPath = Path.Combine(datasetPath, "train-images.idx3-ubyte");
        static string trainLabelsPath = Path.Combine(datasetPath, "train-labels.idx1-ubyte");
        static string testImagesPath = Path.Combine(datasetPath, "t10k-images.idx3-ubyte");
        static string testLabelsPath = Path.Combine(datasetPath, "t10k-labels.idx1-ubyte");

        static void Main(string[] args)
        {
            var config = new
            {
                EpochsCount = 10,
                HiddenLayerSize = 50,
                CrossEntropyError = 0.005f,
                LearningRate = 0.01f
            };
            Stopwatch sw = new Stopwatch();
            sw.Start();

            ImagesContainer trainIC = new ImagesContainer(trainImagesPath);
            LabelsContainer trainLC = new LabelsContainer(trainLabelsPath);
            var imageSize = trainIC.ImagesWidth * trainIC.ImagesHeight;

            Console.WriteLine("Initializing of neural network...");
            NeuralNetwork NN = new NeuralNetwork(imageSize, config.HiddenLayerSize, 10, config.LearningRate);

            var combinedData = CombineImages(trainIC, trainLC);
            Console.WriteLine("Begin training...");
            NN.Train(combinedData, config.EpochsCount, config.CrossEntropyError);
            sw.Stop();
            Console.WriteLine($"Training complete in {sw.ElapsedMilliseconds} milliseconds");
            Console.WriteLine("\r\n______________________________\r\n");

            ImagesContainer testIC = new ImagesContainer(testImagesPath);
            LabelsContainer testLC = new LabelsContainer(testLabelsPath);
            var combinedTestData = CombineImages(testIC, testLC);
            Console.WriteLine("Begin training...");
            NN.Test(combinedTestData, config.CrossEntropyError);

            Console.ReadLine();
        }

        static List<ImageDescription> CombineImages(ImagesContainer trainIC, LabelsContainer trainLC)
        {
            var ret = new List<ImageDescription>();
            for (int i = 0; i < trainIC.ImagesCount; i++)
            {
                ret.Add(new ImageDescription() { Image = trainIC.Images[i], Label = trainLC.Labels[i] });
            }
            return ret;
        }
    }
}
