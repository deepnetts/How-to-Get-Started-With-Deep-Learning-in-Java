package deepnetts.getstarted;

import deepnetts.core.DeepNetts;
import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.examples.util.ExampleDataSets; // ovaj ubaci u 2.0.1 kad dodas jos fixes - popravi sve bugove!!!
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.FileIO;
import deepnetts.util.Tensor;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Recognition of hand-written digits. This example shows how to use
 * convolutional neural network to recognize hand written digits. The problem of
 * hand-written digit recognition is solved as multi-class classification of
 * images
 *
 * Data set description The data set used in this examples is a subset of
 * original MNIST data set, which is considered to be a 'Hello world' for image
 * recognition. The original data set contains 60000 images, dimensions 28x28
 * pixels. Original data set URL: http://yann.lecun.com/exdb/mnist/
 *
 * For the best performance and accuracy the recommended way to run this example
 * is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class HandwrittenDigitRecognition {

    // dimensions of input images
    int imageWidth = 28;
    int imageHeight = 28;

    // training image index and labels
    String labelsFile = "mnist/labels.txt"; // data set ne sme da bud elokalni - neka ga downloaduuje sa github-a - mozda visrec?
    String trainingFile = "mnist/train.txt";

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {

        // download MNIST data set from github
        Path mnistPath = DataSetDownloader.downloadMnistDataSet();
        LOGGER.info("Downloaded MNIST data set to " + mnistPath);

        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.setInvertImages(true);
        LOGGER.info("Loading images...");
        imageSet.loadLabels(new File(labelsFile)); // file with category labels, in this case digits 0-9
        imageSet.loadImages(new File(trainingFile), 1000); // files with list of image paths to use for training,  the second parameter is a number of images in subset of original data set

        ImageSet[] imageSets = imageSet.split(0.65, 0.35); // split data set into training and test sets in given ratio
        ImageSet trainingSet = imageSets[0];
        ImageSet testSet = imageSets[1];
        int labelsCount = imageSet.getLabelsCount(); // the number of image categories/classes, the number of network outputs should correspond to this

        LOGGER.info("Creating neural network architecture...");

        // create convolutional neural network architecture
        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight)
                .addConvolutionalLayer(12, 5)
                .addMaxPoolingLayer(2, 2)
                .addFullyConnectedLayer(60)
                .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                .hiddenActivationFunction(ActivationType.RELU)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        LOGGER.info("Training the neural network");

        // set training options and train the network
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setLearningRate(0.001f) // a percent of error that is used for tuning internal parameters
                .setMaxError(0.05f) // stop training when the specified error threshold is reached
                .setMaxEpochs(1000);  // stop training when  maximum training iterations/epochs is reached
        trainer.train(trainingSet); // run training with the specified training set

        // Test/evaluate trained network to see how it perfroms with unseen data - the test set
        EvaluationMetrics em = neuralNet.test(testSet);

        // print evaluation metrics
        LOGGER.info("Classification metrics");
        LOGGER.info(em);

        // Save trained network to file
        FileIO.writeToFile(neuralNet, "mnistDemo.dnet");

        ExampleImage someImage = new ExampleImage(ImageIO.read(new File("mnist/training/9/00019.png"))); // load some image from file
        someImage.invert(); // used in this example/data set in order to focus on black images and not white background
        Tensor predictions = neuralNet.predict(someImage.getInput()); // get prediction for the specified image
        int maxIdx = indexOfMax(predictions); // get index of prediction with the highest probability
        LOGGER.info(predictions);
        LOGGER.info("Image label with highest probability:"+neuralNet.getOutputLabel(maxIdx));
        
        // shutdown the thread pool
        DeepNetts.shutdown();

    }
    
    int indexOfMax(Tensor tensor) {
        float max = -1;
        int maxIdx = -1;
        for(int i=0; i<tensor.size(); i++) {
            if (tensor.get(i) > max) {
                max = tensor.get(i);
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public static void main(String[] args) throws IOException {
        (new HandwrittenDigitRecognition()).run();
    }
}
