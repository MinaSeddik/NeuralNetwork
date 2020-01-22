package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.Matrix;
import org.javatuples.Pair;
import org.javatuples.Tuple;
import org.javatuples.Unit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Conv2D extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Conv2D.class);

    private Pair<Integer, Integer> kernelSize;
    private String activationFunctionStr;

    public Conv2D(int filters, Tuple inputShape, String activation, Pair<Integer, Integer> kernelSize) {
        this(filters, activation, kernelSize);

        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Unit":
                numOfInputs = ((Unit<Integer>) inputShape).getValue0();
                numOfInputs += 1; // Add 1 for Bias
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
    }

    public Conv2D(int filters, String activation, Pair<Integer, Integer> kernelSize) {
        numOfOutputs = filters;
        activationFunctionStr = activation;
        this.kernelSize = kernelSize;
    }

    @Override
    public String getName() {
        return "conv2d_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        return 0;
    }

    @Override
    public Matrix forwardPropagation(Matrix input) {
        return null;
    }

    @Override
    public void printForwardPropagation(Matrix input) {

    }

    @Override
    public void backPropagation(Matrix costPrime) {

    }

    @Override
    public void updateWeight(double learningRate) {

    }
}
