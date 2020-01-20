package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.activationfunction.SoftMax;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

/**
 * Created by menai on 2019-02-01.
 */
public class CrossEntropyLoss extends LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(CrossEntropyLoss.class);

    @Override
    public double[][] errorCost(double[][] labels, double[][] output) {
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCost received labels: ", labels);
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCost received output: ", output);

//        MatrixManipulator.printMatrix("CrossEntropyLoss calculates errorCost received labels: ", labels);
//        MatrixManipulator.printMatrix("CrossEntropyLoss calculates errorCost received output: ", output);

        // make sure that the both matrices have the same dimension
//        assert (labels != null);
//        assert (output != null);
//        assert (labels.length == output.length);
//        assert (labels[0].length == output[0].length);

        double[][] costs = new double[labels.length][1];

        double y = 0d;
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                y = output[i][j] == 0d ? Double.MIN_VALUE : output[i][j];
//                costs[i][0] += (labels[i][j] * Math.log(output[i][j])) + ((1f - labels[i][j]) * Math.log(1f - output[i][j]));
                costs[i][0] += (labels[i][j] * Math.log(y));// + ((1f - labels[i][j]) * Math.log(1f - y));

            }

//            float deb =  costs[i][0];
//            if( costs[i][0] == Float.POSITIVE_INFINITY ){
//                System.out.println("BEFORE --> //*/*/*//*/*/*/*//*/*/*/*/*/*/*/" + costs[i][0]);
//            }

            costs[i][0] = -costs[i][0];
//            if( costs[i][0] == Float.POSITIVE_INFINITY ){
//                System.out.println("AFTER --> //*/*/*//*/*/*/*//*/*/*/*/*/*/*/before = " + deb + " & after = " + costs[i][0] );
//
//                System.out.println("output --> " + Arrays.toString(output[i]));
//                System.out.println("labels --> " + Arrays.toString(labels[i]));
//                System.exit(-1);
//            }
        }

//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculated costs: ", costs);
//        MatrixManipulator.printMatrix("CrossEntropyLoss calculated costs: ", costs);
        return costs;
    }

    @Override
    public double errorCost(Pair<Vector, Vector> pVector) {
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
    public double accuracy(Pair<Vector, Vector> pVector) {
        double[] labels = pVector.getValue0().asArray();
        double[] output = pVector.getValue1().asArray();

        // Reference: https://kharshit.github.io/blog/2018/12/07/loss-vs-accuracy
        return labels[getMaxProbIndex(output)] == 1d ? 1d : 0d;
    }

    private int getMaxProbIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    public double[][] errorOutputPrime(double[][] labels, double[][] output, ActivationFunction activationFunction) {
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received labels: ", labels);
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received output: ", output);

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
                    logger.debug("CrossEntropyLoss uses SoftMax Activation Function");
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

    @Override
    public double errorCostPrime(Pair<Double, Double> outputPair) {
        double y = outputPair.getValue0();
        double yPrime = outputPair.getValue1();

//        return activationFunction instanceof SoftMax ?
//                yPrime - y : ((-y / yPrime) + ((1 - y) / (1 - yPrime))) * activationFunction.activatePrime(yPrime);

        // assume that the output layer activation function is softmax
//        return ((-y / yPrime) + ((1 - y) / (1 - yPrime)));
        return yPrime - y;
    }

}
