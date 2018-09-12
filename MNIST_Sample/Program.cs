using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Runtime.Api;
using System;


namespace MNIST_Sample
{
    class Program
    {
        static void Main(string[] args)
        {

            // if you are using the code first time rememebr to unzip mnist_test.zip

            var trainingDataLocation = @"data/mnist_train.csv";
            var testDataLocation = @"data/mnist_test.csv";

            var modelEvaluator = new ModelEvaluator();

            var perceptronBinaryModel = new ModelBuilder(trainingDataLocation, new AveragedPerceptronBinaryClassifier()).BuildAndTrain();
            var perceptronBinaryMetrics = modelEvaluator.Evaluate(perceptronBinaryModel, testDataLocation);

            Console.ReadLine();
        }
    }

    public class MNIST_Data
    {
        public MNIST_Data()
        {
            Pixels = new byte[784];
        }

        [Column("0", name: "Number")]
        public byte Number;

        [Column("1", name: "Pixels")]
        public byte[] Pixels;

        [Column(ordinal: "2", name: "Label")]
        public float Label;
    }

    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel;
        public float[] Score;
    }
}
