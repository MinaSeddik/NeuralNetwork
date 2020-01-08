package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.activationfunction.SoftMax;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Created by menai on 2019-02-01.
 */
public class CrossEntropyLoss extends LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(CrossEntropyLoss.class);

    @Override
    public float[][] errorCost(float[][] labels, float[][] output) {
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCost received labels: ", labels);
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCost received output: ", output);

//        MatrixManipulator.printMatrix("CrossEntropyLoss calculates errorCost received labels: ", labels);
//        MatrixManipulator.printMatrix("CrossEntropyLoss calculates errorCost received output: ", output);

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        float[][] costs = new float[labels.length][1];
        MatrixManipulator.initializeMatrix(costs, 0f);

        float y = 0f;
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                y = output[i][j] == 0f ? Float.MIN_VALUE : output[i][j];
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
    public float[][] errorOutputPrime(float[][] labels, float[][] output, ActivationFunction activationFunction) {
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received labels: ", labels);
//        MatrixManipulator.debugMatrix("CrossEntropyLoss calculates errorCostPrime received output: ", output);

        // make sure that the both matrices have the same dimension
        assert (labels != null);
        assert (output != null);
        assert (labels.length == output.length);
        assert (labels[0].length == output[0].length);

        float[][] outputPrime = new float[output.length][output[0].length];
        MatrixManipulator.initializeMatrix(outputPrime, 0f);

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

}
