package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class OutputLayer extends Layer {

    private final static Logger logger = LoggerFactory.getLogger(OutputLayer.class);

    public OutputLayer(String layerName, int numOfInputs, int numOfOutputs, String activationFunctionName, double learningRate) {
        super(layerName, numOfInputs, numOfOutputs, activationFunctionName, learningRate);
    }

    @Override
    public double[][] forwardPropagation() {
//        MatrixManipulator.debugMatrix(layerName + " input:", input);
//        MatrixManipulator.debugMatrix(layerName + " weight:", weight);

        A = MatrixManipulator.multiply(input, weight);
//        MatrixManipulator.debugMatrix(layerName + " A:", A);

        Z = activationFunction.activate(A);
//        MatrixManipulator.debugMatrix(layerName + " Z:", Z);

        return Z;
    }

    @Override
    public void backPropagation(double[][] costOutputPrime) {
        logger.debug("{} BackPropagation calculateDeltaWeight ...", layerName);
        calculateDeltaWeight(costOutputPrime);

        logger.debug("{} BackPropagation prepareErrorCostThenBackPropagate ...", layerName);
        prepareErrorCostThenBackPropagate(costOutputPrime);
    }

    @Override
    public int getNumberOfOutputs() {
        return numOfOutputs;
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

}
