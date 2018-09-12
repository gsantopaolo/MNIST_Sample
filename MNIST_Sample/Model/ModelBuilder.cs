using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;


namespace MNIST_Sample
{
    public class ModelBuilder
    {
        private readonly string _trainingDataLocation;
        private readonly ILearningPipelineItem _algorythm;

        public ModelBuilder(string trainingDataLocation, ILearningPipelineItem algorythm)
        {
            _trainingDataLocation = trainingDataLocation;
            _algorythm = algorythm;
        }

        /// <summary>
        /// Using training data location that is passed trough constructor this method is building
        /// and training machine learning model.
        /// </summary>
        /// <returns>Trained machine learning model.</returns>
        public PredictionModel<MNIST_Data, Prediction> BuildAndTrain()
        {
            var pipeline = new LearningPipeline();
            //pipeline.Add(new TextLoader(_trainingDataLocation).CreateFrom<MNIST_Data>(useHeader: false, separator: ','));
            //pipeline.Add(new MissingValueSubstitutor("FixedAcidity") { ReplacementKind = NAReplaceTransformReplacementKind.Mean});
            var a = CollectionDataSource.Create(helper.ReadMNIST_Data(_trainingDataLocation));
            pipeline.Add(a);


            //pipeline.Add(MakeNormalizer());
            //pipeline.Add(new ColumnConcatenator("Features"));//, "Number", "Pixels"));
            pipeline.Add(_algorythm);

            return pipeline.Train<MNIST_Data, Prediction>();
        }

        private ILearningPipelineItem MakeNormalizer()
        {
            var normalizer = new BinNormalizer();
            normalizer.NumBins = 2;
            normalizer.AddColumn("Label");

            return normalizer;
        }
    }
}
