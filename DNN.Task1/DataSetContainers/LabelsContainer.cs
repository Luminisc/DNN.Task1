using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN.Task1.DataSetContainers
{
    class LabelsContainer
    {
        public int LabelsCount { get; set; }
        public byte[] Labels { get; protected set; }

        protected Stopwatch sw = new Stopwatch();        

        public LabelsContainer(string filepath, bool preload = true)
        {
            sw.Restart();
            using (var file = File.OpenRead(filepath))
            using (BinaryReader reader = new BinaryReader(file))
            {
                // TODO: check magic number vlidity
                // int magicNumber = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                reader.ReadBytes(sizeof(int));
                // TODO: optimize 
                LabelsCount = BitConverter.ToInt32(reader.ReadBytes(sizeof(int)).Reverse().ToArray(), 0);
                Labels = reader.ReadBytes(LabelsCount);
            }
            sw.Stop();
            Console.WriteLine($"Labels file loaded in {sw.ElapsedMilliseconds} ms.");
        }
    }
}
