package com.mina.examples;

import com.mina.examples.mnist.MNistLoader;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.Dense;
import com.mina.ml.neuralnetwork.layer.Model;
import com.mina.ml.neuralnetwork.layer.Sequential;
import com.mina.ml.neuralnetwork.layer.Verbosity;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.javatuples.Unit;

import java.util.List;

public class MNistNeuralNetwork {

    private static final int MNIST_IMAGE_SIZE = 28;

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

        model.fit(xTrain, yTrain, 0.1f, true, 128, 1000, Verbosity.VERBOSE);

        System.out.println("Done Successfully!");

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

