package com.mina.ml.neuralnetwork.activationfunction;

import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by menai on 2019-01-31.
 */
public class SoftMax extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(SoftMax.class);

    @Override
    public double[][] activate(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

//        MatrixManipulator.debugMatrix("SoftMax matrix", matrix);
        for (int i = 0; i < matrix.length; i++) {
            activate(matrix, result, i);
        }

//        MatrixManipulator.debugMatrix("SoftMax result", result);
//        System.exit(0);

        return result;
    }

    private void activate(double[][] matrix, double[][] result, int row) {
        BigDecimal sum = new BigDecimal(0d);

        for (int col = 0; col < matrix[0].length; col++) {
            sum = sum.add(new BigDecimal(Math.exp(matrix[row][col])));
        }

        for (int col = 0; col < matrix[0].length; col++) {
            BigDecimal value = new BigDecimal(activate(matrix[row][col]));
            value = value.divide(sum, 12, RoundingMode.HALF_UP);
            result[row][col] = value.doubleValue();
//            result[row][col] = activate(matrix[row][col]) / sum;
        }
    }

    @Override
    public double activate(double value) {
        return Math.exp(value);
    }

    @Override
    public double activatePrime(double value) {
        // un-defined function
        return 1d;
    }

}
