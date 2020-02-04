package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

// https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
public class Conv2D extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Conv2D.class);

    private Quartet<Integer, Integer, Integer, Integer> inputShape;
    private Pair<Integer, Integer> kernelSize;
    private int filters;

    private D4WeightMatrix weight;
    private D4Matrix deltaWeight;

    private Vector bias;
    private Vector deltaBias;

    private D4Matrix input;
    private D4Matrix A;

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
        return (filters * channels * height * width) + biasSize;
    }

    @Override
    public Tensor forwardPropagation(Tensor inputTensor) {

        // Reference: https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509

//        System.out.println("Conv2D inputShape " + inputShape);


        input = (D4Matrix) inputTensor;
        D4FeatureMatrix features = new D4FeatureMatrix(input.getDimensionCount());
        A = features.buildFeatures(input, weight, kernelSize, filters, getOutputHeight(), getOutputWidth(), bias);
//        System.out.println("A shape = " + A.shape());

        D4Matrix Z = activationFunction.activate(A);
//        System.out.println("Conv2D Z shape = " + Z.shape());

        return Objects.isNull(nextLayer) ? Z : nextLayer.forwardPropagation(Z);
    }

    @Override
    public void printForwardPropagation(Tensor input) {

    }

    @Override
    public void backPropagation(Tensor costPrime) {

        //dE/dZ
        D4Matrix dE_dZ = (D4Matrix) costPrime;
//        System.out.println("dE/dZ shape = " + dE_dZ.shape());

        // dZ/dA
        D4Matrix dZ_dA = activationFunction.activatePrime(A);
//        System.out.println("dZ/dA shape = " + dZ_dA.shape());

        // dE/dZ
        D4Matrix dE_dA = dE_dZ.elementWiseProduct(dZ_dA);
//        System.out.println("dE/dA shape = " + dE_dA.shape());

        // (1) Calculate the gradient for the Bias
        calculateDeltaBias(dE_dA);

        // (2) Calculate the gradient for the Weights
        calculateDeltaWeights(dE_dA);

        if (!Objects.isNull(prevLayer)) {
            D4Matrix cost = calculateOutput(dE_dA);
            prevLayer.backPropagation(cost);
        }

    }

    private D4Matrix calculateOutput(D4Matrix dE_dA) {
        D4FeatureMatrix features = new D4FeatureMatrix(input.getDimensionCount());

        return features.calculateOutputPrime(dE_dA, input, weight, kernelSize, filters, padding);
    }

    private void calculateDeltaWeights(D4Matrix dE_dA) {
        int numberOfSamples = dE_dA.getDimensionCount();

        deltaWeight.reset();
        for (int n = 0; n < numberOfSamples; n++) {
            D3Matrix dEdA = dE_dA.getSubMatrix(n);
            D3Matrix X = input.getSubMatrix(n);
            D4Matrix delta = calculateEntryDeltaWeight(dEdA, X);
            deltaWeight.add(delta);
        }
        // get Average delta-weights
        deltaWeight.divide(numberOfSamples);
    }

    private void calculateDeltaBias(D4Matrix dE_dA) {
        // number of samples
        int numberOfSamples = dE_dA.getDimensionCount();

        double[][] dy = new double[numberOfSamples][filters];
        for (int n = 0; n < numberOfSamples; n++) {
            for (int filter = 0; filter < filters; filter++) {
                dy[n][filter] = dE_dA.getSubMatrix(n, filter).sumAllElements();
            }
        }

        List<Double> avgs = new Matrix(dy).transpose()
                .asVectors()
                .parallelStream()
                .map(v -> v.average())
                .collect(Collectors.toList());
        deltaBias = new Vector(avgs.stream()
                .mapToDouble(Double::doubleValue)
                .toArray());
    }

    private D4Matrix calculateEntryDeltaWeight(D3Matrix dE, D3Matrix X) {
        assert filters == dE.getDepthCount();
        int channels = X.getDepthCount();
        int kernalHeight = kernelSize.getValue0();
        int kernalWidth = kernelSize.getValue1();

        double[][][] x = X.getMatrix();
        double[][][] cost = dE.getMatrix();

        double[][][][] result = new double[filters][channels][kernalHeight][kernalWidth];
        for (int f = 0; f < filters; f++) {
            for (int i = 0; i < kernalHeight; i++) {
                for (int j = 0; j < kernalWidth; j++) {
                    for (int k = 0; k < kernalHeight; k++) {
                        for (int l = 0; l < kernalWidth; l++) {
                            for (int c = 0; c < channels; c++) {
                                result[f][c][i][j] += x[c][i + k][j + l] * cost[f][k][l];
                            }
                        }
                    }
                }
            }
        }

        return new D4Matrix(result);
    }

    @Override
    public void updateWeight(double learningRate) {
        weight.updateWeights(deltaWeight, learningRate);
        if (!Objects.isNull(nextLayer)) {
            nextLayer.updateWeight(learningRate);
        }
    }

    @Override
    public Tensor getWeights() {
        return weight;
    }

    @Override
    public void setWeights(Tensor weight) {
        this.weight = (D4WeightMatrix) weight;
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Quartet<>(inputShape.getValue0(), filters, getOutputHeight(), getOutputWidth());
    }


    private int getOutputHeight() {
        int originalHeight = inputShape.getValue2();
        int kernalHeight = kernelSize.getValue0();
        return 1 + ((originalHeight + 2 * padding - kernalHeight) / strides);
    }

    private int getOutputWidth() {
        int originalWidth = inputShape.getValue3();
        int kernalWidth = kernelSize.getValue1();
        return 1 + ((originalWidth + 2 * padding - kernalWidth) / strides);
    }

}
