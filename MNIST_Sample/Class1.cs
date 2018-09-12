using CsvHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MNIST_Sample
{
    public static class helper
    {
        public static List<MNIST_Data> ReadMNIST_Data(string absolutePath)
        {
            List<MNIST_Data> retVal = new List<MNIST_Data>();

            byte value;
            using (TextReader fileReader = File.OpenText(absolutePath))
            {
                var csv = new CsvReader(fileReader);
                csv.Configuration.HasHeaderRecord = false;
                while (csv.Read())
                {
                    var row = new MNIST_Data();
                    for (int i = 0; csv.TryGetField<byte>(i, out value); i++)
                    {
                        if (i == 0)
                            row.Number = value;
                        else
                            row.Pixels[i - 1] = value;
                    }
                    retVal.Add(row);
                }
            }
            return retVal;
        }
    }
}
