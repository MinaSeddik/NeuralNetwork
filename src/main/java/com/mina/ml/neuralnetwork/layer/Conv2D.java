package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Conv2D extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Conv2D.class);

    private Quartet<Integer, Integer, Integer, Integer> inputShape;
    private Pair<Integer, Integer> kernelSize;
    private int filters;

    private D4WeightMatrix weight;
    private D4WeightMatrix deltaWeight;

    private Vector bias;

    private D4WeightMatrix input;
    private D4WeightMatrix A;
    private D4WeightMatrix Z;

    private String activationFunctionStr;

    // will be constant for now!
    private int padding = 0;
    private int strides = 1;

    public Conv2D(int filters, Tuple inputShape, String activation, Pair<Integer, Integer> kernelSize) {
        this(filters, activation, kernelSize);

        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Quartet":
                this.inputShape = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape for Conv2D Layer");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
    }

    public Conv2D(int filters, String activation, Pair<Integer, Integer> kernelSize) {
        this.filters = filters;
        activationFunctionStr = activation;
        this.kernelSize = kernelSize;

        bias = new Vector(filters);
    }

    @Override
    public void buildupLayer() {

        // init the weight matrix
        int channels = inputShape.getValue1();
        int height = kernelSize.getValue0();
        int width = kernelSize.getValue1();
        weight = new D4WeightMatrix(filters, channels, height, width);
        weight.initializeRandom(-1.0d, 1.0d);

        deltaWeight = new D4WeightMatrix(filters, channels, height, width);

        ActivationFunctionFactory activationFunctionFactory = new ActivationFunctionFactory();
        try {
            activationFunction = activationFunctionFactory.createActivationFunction(activationFunctionStr);
        } catch (Exception ex) {
            logger.error("{}: {}", ex.getClass(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    @Override
    public String getName() {
        return "conv2d_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        int channels = inputShape.getValue1();
        int height = kernelSize.getValue0();
        int width = kernelSize.getValue1();
        int biasSize = bias.size();
        return (filters * channels * height * width ) + biasSize;
    }

    @Override
    public Tensor forwardPropagation(Tensor input) {

        D4Matrix test = (D4Matrix)input;
        System.out.println("forwardPropagation" + test.shape());

//        D4Matrix mat = imagePatches(input, kernelSize);

        System.exit(0);

        return null;
    }

    @Override
    public void printForwardPropagation(Tensor input) {

    }

    @Override
    public void backPropagation(Tensor costPrime) {

    }

    @Override
    public void updateWeight(double learningRate) {

    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public void setWeights(Tensor weight) {

    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Quartet<Integer, Integer, Integer, Integer>)inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Quartet<>(inputShape.getValue0(), filters, getOutputHeight(), getOutputWidth());
    }


    private int getOutputHeight() {
        int originalHeight = inputShape.getValue2();
        int kernalHeight = kernelSize.getValue0();
        return 1 + ((originalHeight + 2*padding - kernalHeight) / strides);
    }

    private int getOutputWidth() {
        int originalWidth = inputShape.getValue3();
        int kernalWidth = kernelSize.getValue1();
        return 1 + ((originalWidth + 2*padding - kernalWidth) / strides);
    }

}
