package com.mina.examples;

import com.mina.examples.mnist.MNistLoader;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class MNistConvolutionalNeuralNetwork2 {

    private static final int MNIST_IMAGE_CHANNELS = 1;
    private static final int MIST_IMAGE_HEIGHT = 28;
    private static final int MNIST_IMAGE_WIDTH = 28;

    private static final String MNIST_DATA_DIR = "mnist/models/";
    private static final String MNIST_MODEL_FILE = "cnn_model.bin";
    private static String fileName = "cnn_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.bin";

    public static void main(String[] args) {

        MNistLoader mnistLoader = new MNistLoader();
        Quartet<List<double[][][]>, List<double[]>, List<double[][][]>, List<double[]>> dataset = mnistLoader.loadMNistDataSet2();
        List<double[][][]> xTrain = dataset.getValue0();
        List<double[]> yTrain = dataset.getValue1();
        List<double[][][]> xTest = dataset.getValue2();
        List<double[]> yTest = dataset.getValue3();

        // Normalize xTrain and xTest, Divide them by 255
        normalize(xTrain);
        normalize(xTest);

        // save the model
        String dirPath = MNistConvolutionalNeuralNetwork2.class.getClassLoader()
                .getResource(MNIST_DATA_DIR)
                .getFile();
        String modelFilePath = new File(dirPath, MNIST_MODEL_FILE).getAbsolutePath();

        Model model = Model.load(modelFilePath);
        model.summary(line -> System.out.println(line));

//        String filePath = new File(dirPath, fileName).getAbsolutePath();
//        List<ModelCheckpoint> callbacksList = Arrays.asList(new ModelCheckpoint(filePath));
        model.fit(xTrain, yTrain, 0.1f, true, 128, 300, Verbosity.VERBOSE, null);

        Pair<Double, Double> testStats = model.evaluate(xTest, yTest);
        double test_acc = testStats.getValue1();
        System.out.println(String.format("Test accuracy: %.2f%%", (test_acc * 100)));

        // finally save back the model after training
        model.save(modelFilePath);
    }

    private static void normalize(List<double[][][]> list) {
        list.stream().forEach(array -> normalize(array));
    }

    private static void normalize(double[][][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                for (int k = 0; k < array[i][j].length; k++) {
                    array[i][j][k] /= 255d;
                }
            }
        }
    }
}

