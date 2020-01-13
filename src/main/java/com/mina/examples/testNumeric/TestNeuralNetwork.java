package com.mina.examples.testNumeric;

import com.mina.ml.neuralnetwork.Configuration;
import com.mina.ml.neuralnetwork.Constants;
import com.mina.ml.neuralnetwork.NeuralNetwork;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by menai on 2019-02-13.
 */
public class TestNeuralNetwork {

    private static final int DATA_SET_SIZE = 1;


    public static void main(String[] args) {

        // generate 1000000 number
        List<Float> list = generateRandomNumbers();
        double[][] inputDataSet = new double[DATA_SET_SIZE][2];
        double[][] output = new double[DATA_SET_SIZE][2];

        for (int i = 0; i < DATA_SET_SIZE; i++) {
            inputDataSet[i][0] = 0.05d;
            inputDataSet[i][1] = 0.1d;

            output[i][0] = 0.01d;
            output[i][1] = 0.99d;
        }


        Properties neuralNetworkProperties = new Properties();

        neuralNetworkProperties.put(Constants.NUMBER_OF_FEATURES, 2);

        neuralNetworkProperties.put(Constants.NUMBER_OF_OUTPUT_NODES, 2);
        neuralNetworkProperties.put(Constants.OUTPUT_ACTIVATION_FUNCTION, "softmax");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYERS, 1);

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "1"), 3);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "1"), "sigmoid");



        neuralNetworkProperties.put(Constants.LEARNING_RATE, 0.1);

        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "CrossEntropyLoss");

        neuralNetworkProperties.put(Constants.BATCH_SIZE, 2);
        neuralNetworkProperties.put(Constants.MAX_EPOCH, 13);

        Configuration configuration = new Configuration(neuralNetworkProperties);
        try {
            NeuralNetwork neuralNetwork = new NeuralNetwork(configuration);

//            double[][] labels = MatrixManipulator.vectorToMatrix(output);
//            double[][] labels = MatrixManipulator.transposeMatrix(output);
            double[][] labels = output;

            neuralNetwork.fetchDataSet(inputDataSet, labels);
            neuralNetwork.train();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static List<Float> generateRandomNumbers() {
        int rangeMin = -DATA_SET_SIZE;
        int rangeMax = DATA_SET_SIZE;

        Set<Integer> X = new HashSet<>();
        Random random = new Random();
        int temp;
        int count = 0;

        while (count < DATA_SET_SIZE) {
            temp = rangeMin + (rangeMax - rangeMin) * random.nextInt();

            if (!X.contains(temp)) {
                X.add(temp);
                count++;
            }

        }

        return X.stream()
                .map(i -> i * 1f)
                .collect(Collectors.toList());
    }
}
