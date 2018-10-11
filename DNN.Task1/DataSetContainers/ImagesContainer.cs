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
        public List<float[]> Images { get; protected set; }

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

                Images = Enumerable.Range(0, ImagesCount)
                    .Select(x => reader.ReadBytes(imageSize))
                    .Select(x => x.Select(y => y / 255.0f).ToArray())
                    .ToList();
            }
            sw.Stop();
            Console.WriteLine($"Images file loaded in {sw.ElapsedMilliseconds} ms.");
        }
    }
}
