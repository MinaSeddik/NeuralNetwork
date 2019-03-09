package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class SoftMax extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(SoftMax.class);

    @Override
    public float[][] activate(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            activate(matrix, result, i);
        }
        return result;
    }

    private void activate(float[][] matrix, float[][] result, int row) {
        float sum = 0f;
        for (int col = 0; col < matrix[0].length; col++) {
            sum += Math.exp(matrix[row][col]);
        }

        for (int col = 0; col < matrix[0].length; col++) {
            result[row][col] = activate(matrix[row][col]) / sum;
        }
    }

    @Override
    public float activate(float value) {
        return (float) Math.exp(value);
    }

    @Override
    public float activatePrime(float value) {
        // un-defined function
        return 0f;
    }

}
