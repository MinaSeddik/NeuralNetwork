package com.mina.examples;

import com.mina.examples.mnist.MNistLoader;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.*;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.javatuples.Unit;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MNistNeuralNetwork3 {

    private static final String MNIST_DATA_DIR = "mnist/models/";
    private static final String MNIST_MODEL_FILE = "nn_model.bin";
    private static final String MNIST_WEIGHT_FILE = "nn_best-weights.bin";

    public static final int NUMBER_OF_TEST_SAMPLES = 10;

    public static void main(String[] args) {

        String dirPath = MNistNeuralNetwork2.class.getClassLoader()
                .getResource(MNIST_DATA_DIR)
                .getFile();

        MNistLoader mnistLoader = new MNistLoader();
        Quartet<List<double[]>, List<double[]>, List<double[]>, List<double[]>> dataset = mnistLoader.loadMNistDataSet();
        List<double[]> xTrain = dataset.getValue0();
        List<double[]> yTrain = dataset.getValue1();
        List<double[]> xTest = dataset.getValue2();
        List<double[]> yTest = dataset.getValue3();

        // Normalize xTrain and xTest, Divide them by 255
        normalize(xTrain);
        normalize(xTest);


        // load the model
        String modelFilePath = new File(dirPath, MNIST_MODEL_FILE).getAbsolutePath();
        Model model = Model.load(modelFilePath);
        model.summary(line -> System.out.println(line));

        // load the best weight
        model.loadWeights(MNIST_WEIGHT_FILE);

        model.evaluate(xTest, yTest);

        // get Random 10 samples from the test
        Random random = new Random();
        System.out.println(String.format("ID\tActual\tPredicted"));
        for(int i=0;i<NUMBER_OF_TEST_SAMPLES;i++){
            int r = random.nextInt(xTest.size());
            double[] inputData = xTest.get(r);
            double[] label = yTest.get(r);
            Vector v = new Vector(label);

            List<Integer> results = model.predictClasses(Arrays.asList(inputData));
            System.out.println(String.format("%d\t%d\t%d",(i+1), v.argMaxIndex(), results.get(0)));
        }




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
