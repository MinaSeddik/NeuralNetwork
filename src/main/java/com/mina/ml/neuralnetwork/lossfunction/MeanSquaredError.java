package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Created by menai on 2019-02-01.
 */
public class MeanSquaredError extends LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(MeanSquaredError.class);

    @Override
    public double[][] errorCost(double[][] labels, double[][] output) {
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

        double[][] costs = new double[labels.length][1];
        MatrixManipulator.initializeMatrix(costs, 0d);

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
    public double errorCost(Pair<Vector, Vector> pVector){
        double[] y = pVector.getValue0().asArray();
        double[] yPrime = pVector.getValue1().asArray();

        return IntStream.range(0, y.length)
                .mapToDouble(i -> Math.pow(y[i] - yPrime[i], 2d)/2d).sum();
    }

    @Override
    public double[][] errorOutputPrime(double[][] labels, double[][] output, ActivationFunction activationFunction) {
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCostPrime received labels: ", labels);
//        MatrixManipulator.debugMatrix("MeanSquaredError calculates errorCostPrime received output: ", output);

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        double[][] outputPrime = new double[output.length][output[0].length];
        MatrixManipulator.initializeMatrix(outputPrime, 0d);

        for (int i = 0; i < outputPrime.length; i++) {
            for (int j = 0; j < outputPrime[0].length; j++) {
                outputPrime[i][j] = (output[i][j] - labels[i][j]) * activationFunction.activatePrime(output[i][j]);
            }
        }

//        MatrixManipulator.debugMatrix("MeanSquaredError calculated costPrimes: ", outputPrime);
        return outputPrime;
    }


}
