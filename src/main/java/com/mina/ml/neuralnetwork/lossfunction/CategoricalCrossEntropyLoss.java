package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.activationfunction.SoftMax;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

public class CategoricalCrossEntropyLoss extends LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(CategoricalCrossEntropyLoss.class);

    @Override
    public double[][] errorCost(double[][] labels, double[][] output) {
        double[][] costs = new double[labels.length][1];

        double yPrime;
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                yPrime = output[i][j] == 0d ? Double.MIN_VALUE : output[i][j];
                costs[i][0] += (labels[i][j] * Math.log(yPrime));
            }
            costs[i][0] = -costs[i][0];
        }

        return costs;
    }

    @Override
    public double errorCost(Pair<Vector, Vector> pVector){
        double[] y = pVector.getValue0().asArray();
        double[] yPrime = pVector.getValue1().asArray();

        double sum = IntStream.range(0, y.length)
                .mapToDouble(i -> {
                    double yPrimeVal = yPrime[i] == 0d ? Double.MIN_VALUE : yPrime[i];
                    return y[i] * Math.log(yPrimeVal);
                }).sum();

        return -sum;
    }

    @Override
    public double[][] errorOutputPrime(double[][] labels, double[][] output, ActivationFunction activationFunction) {
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received labels: ", labels);
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received output: ", output);


        // please revisit this code

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        double[][] outputPrime = new double[output.length][output[0].length];
        MatrixManipulator.initializeMatrix(outputPrime, 0d);

        for (int i = 0; i < outputPrime.length; i++) {
            for (int j = 0; j < outputPrime[0].length; j++) {
                if (activationFunction instanceof SoftMax) {
                    logger.debug("BinaryCrossEntropyLoss uses SoftMax Activation Function");
                    outputPrime[i][j] = output[i][j] - labels[i][j];
                } else {
                    outputPrime[i][j] = ((-labels[i][j] / output[i][j]) + ((1 - labels[i][j]) / (1 - output[i][j])))
                            * activationFunction.activatePrime(output[i][j]);
                }
            }
        }

//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculated costPrimes: ", outputPrime);
        return outputPrime;
    }

}