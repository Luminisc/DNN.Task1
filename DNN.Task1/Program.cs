using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DNN.Task1
{
    class Program
    {
        static string mnistFilepath = @"..\..\..\DataSet\train-images.idx3-ubyte";
        static void Main(string[] args)
        {
            if (args.Length > 0) mnistFilepath = args[0];
            Stopwatch sw = new Stopwatch();
            using (BinaryReader reader = new BinaryReader(File.OpenRead(mnistFilepath)))
            {
                int magicNumber = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                int imagesCount = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                int imageWidth = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                int imageHeight = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                int imageSize = imageWidth * imageHeight;

                //// just for doing anything
                int[] arr = new int[imagesCount];
                int a = 0;
                sw.Start();
                for (int i = 0; i < imagesCount; i++)
                {
                    var image = reader.ReadBytes(imageSize);
                    arr[a++] = Sum(image);
                }
                sw.Stop();

                Console.WriteLine($"Elapsed: {sw.ElapsedMilliseconds} milliseconds");
                Console.ReadLine();
            }
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int Sum(byte[] arr)
        {
            int result = 0;
            foreach (var item in arr)
            {
                result += item;
            }
            return result;
        }
    }
}
