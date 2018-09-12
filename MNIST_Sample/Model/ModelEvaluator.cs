using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;


namespace MNIST_Sample
{
    public class ModelEvaluator
    {
        /// <summary>
        /// Ussing passed testing data and model, it calculates model's accuracy.
        /// </summary>
        /// <returns>Accuracy of the model.</returns>
        public BinaryClassificationMetrics Evaluate(PredictionModel<MNIST_Data, Prediction> model, string testDataLocation)
        {
            var testData = new TextLoader(testDataLocation).CreateFrom<MNIST_Data>(useHeader: false, separator: ',');
            var metrics = new BinaryClassificationEvaluator().Evaluate(model, testData);
            return metrics;
        }
    }
}
