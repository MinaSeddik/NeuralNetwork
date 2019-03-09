package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(ActivationFunction.class);

    public float[][] activate(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activate(matrix[i][j]);
            }
        }

        return result;
    }

    public float[][] activatePrime(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activatePrime(matrix[i][j]);
            }
        }

        return result;
    }

    public abstract float activate(float value);

    public abstract float activatePrime(float value);
}
