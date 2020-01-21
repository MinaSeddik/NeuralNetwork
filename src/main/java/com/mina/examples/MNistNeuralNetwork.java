package com.mina.examples;

import com.mina.examples.mnist.MNistLoader;
import com.mina.examples.mnist.MNistTraining;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.javatuples.Unit;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class MNistNeuralNetwork {

    private static final int MNIST_IMAGE_SIZE = 28;

    private static final String MNIST_DATA_DIR = "mnist/models/";
    private static final String MNIST_MODEL_FILE = "model.bin";

    public static void main(String[] args) {

        MNistLoader mnistLoader = new MNistLoader();
        Quartet<List<double[]>, List<double[]>, List<double[]>, List<double[]>> dataset = mnistLoader.loadMNistDataSet();
        List<double[]> xTrain = dataset.getValue0();
        List<double[]> yTrain = dataset.getValue1();
        List<double[]> xTest = dataset.getValue2();
        List<double[]> yTest = dataset.getValue3();

        // Normalize xTrain and xTest, Divide them by 255
        normalize(xTrain);
        normalize(xTest);

        // Build Neural Network
        Tuple inputShape = new Unit(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE);
        Model model = new Sequential(new Dense[]{
                new Dense(64, inputShape, "relu"),
                new Dense(64, "relu"),
                new Dense(10, "softmax")
        });

        model.summary(line -> System.out.println(line));

        double learningRate = 0.001;
        Optimizer optimizer = new Optimizer(learningRate);
        model.compile(optimizer, "categorical_crossentropy", "");

        String filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5";
        ModelCheckpoint checkpoint = new ModelCheckpoint(filepath);
        List<ModelCheckpoint> callbacksList = Arrays.asList(checkpoint);

        model.fit(xTrain, yTrain, 0.1f, true, 128, 300,
                Verbosity.VERBOSE, callbacksList);

        Pair<Double, Double> testStats = model.evaluate(xTest, yTest);
        double test_acc = testStats.getValue1();
        System.out.println(String.format("Test accuracy: %.2f%%", (test_acc * 100)));

        // save the model
        String dirPath = MNistNeuralNetwork2.class.getClassLoader()
                .getResource(MNIST_DATA_DIR)
                .getFile();
        String modelFilePath = new File(dirPath, MNIST_MODEL_FILE).getAbsolutePath();

        model.save(modelFilePath);

        System.out.println("After loading ...");
        Model loadedModel = Model.load(modelFilePath);
        loadedModel.summary(line -> System.out.println(line));
        loadedModel.fit(xTrain, yTrain, 0.1f, true, 128, 10, Verbosity.VERBOSE, null);
        loadedModel.save(modelFilePath);
    }

    private static void normalize(List<double[]> list) {
        list.stream().forEach(array -> normalize(array));
    }

    private static void normalize(double[] array) {
        for(int i=0;i<array.length;i++){
            array[i]/= 255d;
        }
    }
}

