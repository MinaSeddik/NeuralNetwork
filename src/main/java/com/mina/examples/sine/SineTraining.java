package com.mina.examples.sine;

import com.mina.ml.neuralnetwork.Configuration;
import com.mina.ml.neuralnetwork.Constants;
import com.mina.ml.neuralnetwork.NeuralNetwork;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by menai on 2019-02-13.
 */
public class SineTraining {

    private static final int DATA_SET_SIZE = 10000;


    public static void main(String[] args) {

        // generate 1000000 number
        List<Float> list = generateRandomNumbers();
        float[][] inputDataSet = new float[DATA_SET_SIZE][5];
        float[] output = new float[DATA_SET_SIZE];

        for (int i = 0; i < DATA_SET_SIZE; i++) {
            inputDataSet[i][0] = list.get(i);
            inputDataSet[i][1] = (float) Math.pow(list.get(i), 2);
            inputDataSet[i][2] = 2 * list.get(i);
            inputDataSet[i][3] = (float) Math.sqrt(Math.abs(list.get(i)));
            inputDataSet[i][4] =  list.get(i) / 2.0f;

            output[i] = (float) Math.sin(list.get(i));
        }

        // just for test
//        for (int i = 0; i < 1000; i++) {
//            System.out.println(inputDataSet[i][0] + " " + inputDataSet[i][1] + " "+ inputDataSet[i][2]
//                    + " " + inputDataSet[i][3] + " " + inputDataSet[i][4] + " -> " + output[i]);
//        }

        Properties neuralNetworkProperties = new Properties();

        neuralNetworkProperties.put(Constants.NUMBER_OF_FEATURES, 5);

        neuralNetworkProperties.put(Constants.NUMBER_OF_OUTPUT_NODES, 1);
        neuralNetworkProperties.put(Constants.OUTPUT_ACTIVATION_FUNCTION, "tanh");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYERS, 2);

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "1"), 16);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "1"), "sigmoid");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "2"), 32);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "2"), "relu");


        neuralNetworkProperties.put(Constants.LEARNING_RATE, 0.001);

        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "MeanSquaredError");

        neuralNetworkProperties.put(Constants.BATCH_SIZE, 256);
        neuralNetworkProperties.put(Constants.MAX_EPOCH, 1000000);

        Configuration configuration = new Configuration(neuralNetworkProperties);
        try {
            NeuralNetwork neuralNetwork = new NeuralNetwork(configuration);

            float[][] labels = MatrixManipulator.vectorToMatrix(output);
            labels = MatrixManipulator.transposeMatrix(labels);

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
