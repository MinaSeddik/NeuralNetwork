package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-02-01.
 */
public class MeanSquaredError extends LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(MeanSquaredError.class);

    @Override
    public float[][] errorCost(float[][] labels, float[][] output) {
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCost received labels: ", labels);
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCost received output: ", output);
//
//        MatrixManipulator.printMatrix("MeanSquaredError calculates errorCost received labels: ", labels);
//        MatrixManipulator.printMatrix("MeanSquaredError calculates errorCost received output: ", output);

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        float[][] costs = new float[labels.length][1];
        MatrixManipulator.initializeMatrix(costs, 0f);

        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                costs[i][0] += (Math.pow((labels[i][j] - output[i][j]), 2d)) / 2;
            }
        }

//        MatrixManipulator.debugMatrix("MeanSquaredError calculated costs: ", costs);
//        MatrixManipulator.printMatrix("MeanSquaredError calculated costs: ", costs);
        return costs;
    }

    @Override
    public float[][] errorOutputPrime(float[][] labels, float[][] output, ActivationFunction activationFunction) {
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCostPrime received labels: ", labels);
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCostPrime received output: ", output);

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        float[][] outputPrime = new float[output.length][output[0].length];
        MatrixManipulator.initializeMatrix(outputPrime, 0f);

        for (int i = 0; i < outputPrime.length; i++) {
            for (int j = 0; j < outputPrime[0].length; j++) {
                outputPrime[i][j] = (output[i][j] - labels[i][j]) * activationFunction.activatePrime(output[i][j]);
            }
        }

//        MatrixManipulator.debugMatrix("MeanSquaredError calculated costPrimes: ", outputPrime);
        return outputPrime;
    }


}
