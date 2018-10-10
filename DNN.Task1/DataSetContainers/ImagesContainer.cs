using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN.Task1.DataSetContainers
{
    class ImagesContainer
    {
        public int ImagesCount { get; set; }
        public int ImagesWidth { get; set; }
        public int ImagesHeight { get; set; }

        protected Stopwatch sw = new Stopwatch();
        protected float[] images;

        public ImagesContainer(string filepath, bool preload = true)
        {
            sw.Restart();
            using (var file = File.OpenRead(filepath))
            using (BinaryReader reader = new BinaryReader(file))
            {
                // TODO: check magic number validity & implement faster int reverter
                // int magicNumber = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                reader.ReadBytes(sizeof(int));
                // TODO: optimize 
                ImagesCount = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                ImagesWidth = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                ImagesHeight = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                var imageSize = ImagesWidth * ImagesHeight;

                var buffer = reader.ReadBytes(imageSize * ImagesCount);
                images = buffer.Select(x => x / 255.0f).ToArray();
            }
            sw.Stop();
            Console.WriteLine($"Images file loaded in {sw.ElapsedMilliseconds} ms.");
        }


    }
}
