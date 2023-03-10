package io.github.nickid2018.ipgs;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.io.File;

public class MNISTFeedforwardTest {

    @Test
    public void main() throws Exception {
        System.setProperty("org.bytedeco.javacpp.maxPhysicalBytes", "3G");

        SameDiff sd = SameDiff.create();

        //Properties for MNIST dataset:
        int nIn = 28 * 28;
        int nOut = 10;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

        //Define hidden layer - MLP (fully connected)
        int layerSize0 = 128;
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', nIn, layerSize0), DataType.FLOAT, nIn, layerSize0);
        SDVariable b0 = sd.zero("b0", 1, layerSize0);
        SDVariable activations0 = sd.nn().tanh(in.mmul(w0).add(b0));

        //Define output layer - MLP (fully connected) + softmax
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', layerSize0, nOut), DataType.FLOAT, layerSize0, nOut);
        SDVariable b1 = sd.zero("b1", 1, nOut);

        SDVariable z1 = activations0.mmul(w1).add("prediction", b1);
        SDVariable softmax = sd.nn().softmax("softmax", z1);

        //Define loss function:
        SDVariable diff = sd.math.squaredDifference(softmax, label);
        SDVariable lossMse = diff.mean();

        sd.setLossVariables(lossMse);

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();

        sd.setTrainingConfig(config);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);

        //Perform training for 2 epochs
        int numEpochs = 2;
        sd.fit(trainData, numEpochs);

        //Evaluate on test set:
        String outputVariable = "softmax";
        Evaluation evaluation = new Evaluation();
        sd.evaluate(testData, outputVariable, evaluation);

        //Print evaluation statistics:
        System.out.println(evaluation.stats());

        //Save the trained network for inference - FlatBuffers format
        File saveFileForInference = new File("sameDiffExampleInference.fb");
        sd.asFlatFile(saveFileForInference);

        SameDiff loadedForInference = SameDiff.fromFlatFile(saveFileForInference);

        //Perform inference on restored network
        INDArray example = new MnistDataSetIterator(1, false, 12345).next().getFeatures();
        loadedForInference.getVariable("input").setArray(example);
        INDArray output = loadedForInference.getVariable("softmax").eval();

        System.out.println("-----------------------");
        System.out.println(example.reshape(28, 28));
        System.out.println("Output probabilities: " + output);
        System.out.println("Predicted class: " + output.argMax().getInt(0));
    }

}
