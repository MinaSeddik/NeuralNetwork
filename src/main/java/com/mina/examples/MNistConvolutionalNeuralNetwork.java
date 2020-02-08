package com.mina.examples;

import com.mina.examples.mnist.MNistLoader;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;

import java.io.File;
import java.util.List;

public class MNistConvolutionalNeuralNetwork {

    private static final int MNIST_IMAGE_CHANNELS = 1;
    private static final int MIST_IMAGE_HEIGHT = 28;
    private static final int MNIST_IMAGE_WIDTH = 28;

    private static final String MNIST_DATA_DIR = "mnist/models/";
    private static final String MNIST_MODEL_FILE = "cnn_model.bin";
    private static String fileName = "cnn_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.bin";

    public static void main(String[] args) {

        String dirPath = MNistNeuralNetwork2.class.getClassLoader()
                .getResource(MNIST_DATA_DIR)
                .getFile();

        MNistLoader mnistLoader = new MNistLoader();
        Quartet<List<double[][][]>, List<double[]>, List<double[][][]>, List<double[]>> dataset = mnistLoader.loadMNistDataSet2();
        List<double[][][]> xTrain = dataset.getValue0();
        List<double[]> yTrain = dataset.getValue1();
        List<double[][][]> xTest = dataset.getValue2();
        List<double[]> yTest = dataset.getValue3();

        // Normalize xTrain and xTest, Divide them by 255
        normalize(xTrain);
        normalize(xTest);

        // Build Convolution Neural Network
        Tuple inputShape = new Quartet<>(0, MNIST_IMAGE_CHANNELS, MIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH);
        Pair<Integer, Integer> kernal = new Pair<>(3, 3);
        Pair<Integer, Integer> poolsize = new Pair<>(2, 2);

        Model model = new Sequential();
        model.add(new Conv2D(32, inputShape, "relu", kernal));
        model.add(new MaxPooling2D(poolsize));
        model.add(new Flatten());
        model.add(new Dense(64, "relu"));
        model.add(new Dense(10, "softmax"));

        model.summary(line -> System.out.println(line));

        double learningRate = 0.1;
//        double learningRate = 0.001;
        Optimizer optimizer = new Optimizer(learningRate);
        model.compile(optimizer, "categorical_crossentropy", "");

//        String filePath = new File(dirPath, fileName).getAbsolutePath();
//        List<ModelCheckpoint> callbacksList = Arrays.asList(new ModelCheckpoint(filePath));
        List<ModelCheckpoint> callbacksList = null;

//        System.out.println(String.format("Total Memory: %.4f Gigs" ,Runtime.getRuntime().totalMemory()/(1024d*1024d*1024d)));
        model.fit(xTrain, yTrain, 0.1f, true, 128, 300,
                Verbosity.VERBOSE, callbacksList);

        Pair<Double, Double> testStats = model.evaluate(xTest, yTest);
        double test_acc = testStats.getValue1();
        System.out.println(String.format("Test accuracy: %.2f%%", (test_acc * 100)));

        // save the model
        String modelFilePath = new File(dirPath, MNIST_MODEL_FILE).getAbsolutePath();
        model.save(modelFilePath);

        // load the model
        Model loadedModel = Model.load(modelFilePath);
        loadedModel.summary(line -> System.out.println(line));

        System.out.println("Total Memory:" + Runtime.getRuntime().totalMemory());
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

