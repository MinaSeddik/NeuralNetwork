package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class InputLayer extends Layer2 {

    private final static Logger logger = LoggerFactory.getLogger(InputLayer.class);

    public InputLayer(String layerName, int numOfFeatures) {
        super(layerName, numOfFeatures);
    }

    @Override
    public double[][] forwardPropagation() {
        // add column of ones in the very beginning of the matrix

//        MatrixManipulator.debugMatrix("before add onces: ", input);

//        System.out.println(String.format("Input input = (%d, %d)", input.length, input[0].length));
        double[][] outMatrix = MatrixManipulator.addColumnOfOnes(input);
//        System.out.println(String.format("Input, After adding bias, input = (%d, %d)", outMatrix.length, outMatrix[0].length));
//        MatrixManipulator.debugMatrix("after add onces: ", outMatrix);
//        System.exit(0);
//        MatrixManipulator.debugMatrix(layerName + " input [*After Adding Column of Ones*]:", outMatrix);

        return nextLayer.input(outMatrix)
                .forwardPropagation();
    }

    @Override
    public void backPropagation(double[][] costOutputPrime) {
        /* UnApplicable */
    }

    @Override
    public int getNumberOfOutputs() {
        return numOfInputs + 1;  // far bias
    }

    @Override
    public ActivationFunction getActivationFunction() {
        /* UnApplicable */
        return null;
    }
}
