package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class HiddenLayer extends Layer {

    private final static Logger logger = LoggerFactory.getLogger(HiddenLayer.class);

    public HiddenLayer(String layerName, int numOfInputs, int numOfOutputs, String activationFunctionName, double learningRate) {
        super(layerName, numOfInputs, numOfOutputs, activationFunctionName, learningRate);
    }

    @Override
    public float[][] forwardPropagation() {
//        MatrixManipulator.debugMatrix(layerName + " input:", input);
//        MatrixManipulator.debugMatrix(layerName + " weight:", weight);

        A = MatrixManipulator.multiply(input, weight);
//        MatrixManipulator.debugMatrix(layerName + " A:", A);

        Z = activationFunction.activate(A);
//        MatrixManipulator.debugMatrix(layerName + " Z:", Z);

        // add column of ones in the very beginning of the matrix
        float[][] outMatrix = MatrixManipulator.addColumnOfOnes(Z);
//        MatrixManipulator.debugMatrix(layerName + " input [*After Adding Column of Ones*]:", outMatrix);

        return nextLayer.input(outMatrix)
                .forwardPropagation();
    }

    @Override
    public void backPropagation(float[][] prevCostPrime) {
//logger.info("HERE 7.1");
        float[][] primeA = activationFunction.activatePrime(A);
//        MatrixManipulator.debugMatrix(layerName + " A prime: ", primeA);
//logger.info("HERE 7.2");
//        MatrixManipulator.debugMatrix(layerName + " prevCostPrime: ", prevCostPrime);
        float[][] costOutputPrime = MatrixManipulator.multiplyEntries(prevCostPrime, primeA);
//        MatrixManipulator.debugMatrix(layerName + " costOutputPrime: ", costOutputPrime);
//logger.info("HERE 7.3");
        logger.debug("{} BackPropagation calculateDeltaWeight ...", layerName);
        calculateDeltaWeight(costOutputPrime);
//logger.info("HERE 7.4");
        logger.debug("{} BackPropagation prepareErrorCostThenBackPropagate ...", layerName);
        prepareErrorCostThenBackPropagate(costOutputPrime);
    }

    @Override
    public int getNumberOfOutputs() {
        return numOfOutputs + 1;  // for bias
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

}
