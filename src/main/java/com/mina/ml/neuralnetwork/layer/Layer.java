package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class Layer {

    private final static Logger logger = LoggerFactory.getLogger(Layer.class);

    protected String layerName;

    protected int numOfInputs;
    protected int numOfOutputs;

    protected Layer previousLayer;
    protected Layer nextLayer;

    protected double[][] input;
    protected double[][] A;
    protected double[][] Z;

    protected double[][] weight;
    protected double[][] deltaWeight;

    protected ActivationFunction activationFunction;
    protected double learningRate;

    public Layer(String layerName, int numOfInputs, int numOfOutputs, String activationFunctionName, double learningRate) {
        this.layerName = layerName;

        this.numOfInputs = numOfInputs;
        this.numOfOutputs = numOfOutputs;

        this.learningRate = learningRate;

        // init the weight matrix
        weight = new double[numOfInputs][numOfOutputs];
        initializeWeights(weight);

        deltaWeight = new double[numOfInputs][numOfOutputs];

        ActivationFunctionFactory activationFunctionFactory = new ActivationFunctionFactory();
        try {
            activationFunction = activationFunctionFactory.createActivationFunction(activationFunctionName);
        } catch (Exception ex) {
            logger.error("Can't create Activation Function, Exception: {}", ex);
        }
    }

    /* for Input layer */
    public Layer(String layerName, int numOfInputs) {
        this.layerName = layerName;
        this.numOfInputs = numOfInputs;
    }

    private void initializeWeights(double[][] weight) {
        double rangeMin = -1.0d;
        double rangeMax = 1.0d;
        Random r = new Random();
//        r.setSeed(100);


        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", rangeMin, rangeMax));
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[0].length; j++) {
                weight[i][j] = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
            }
        }
    }

    protected void calculateDeltaWeight(double[][] costOutputPrime) {
        // make sure that both of them has the same row ( # of examples)
        assert (costOutputPrime.length == input.length);

        assert (deltaWeight.length == input[0].length);
        assert (deltaWeight[0].length == costOutputPrime.length);

        // reset delta weights
        MatrixManipulator.initializeMatrix(deltaWeight, 0d);

        for (int i = 0; i < input.length; i++) {
//            logger.info("HERE 7.3.1");
            double[][] in = MatrixManipulator.transposeMatrix(MatrixManipulator.vectorToMatrix(input[i]));
            double[][] out = MatrixManipulator.vectorToMatrix(costOutputPrime[i]);
//            logger.info("HERE 7.3.2");
//            MatrixManipulator.debugMatrix(layerName + " in: ", in);
//            MatrixManipulator.debugMatrix(layerName + " out: ", out);
//            logger.info("HERE 7.3.3");
            double[][] dWeights = MatrixManipulator.multiply(in, out);
//            MatrixManipulator.debugMatrix(layerName + " dWeights: ", dWeights);
//            logger.info("HERE 7.3.4");
            accumulateDeltaWeights(dWeights);
//            logger.info("HERE 7.3.5");
        }

        // get mean weights
        for (int i = 0; i < deltaWeight.length; i++) {
            for (int j = 0; j < deltaWeight[0].length; j++) {
                deltaWeight[i][j] /= costOutputPrime.length; // number of examples
            }
        }
//        MatrixManipulator.debugMatrix(layerName + " deltaWeight: ", deltaWeight);
    }

    protected void prepareErrorCostThenBackPropagate(double[][] costOutputPrime) {
        double[][] costPrime = null;

        if (previousLayer instanceof HiddenLayer) {

            // prepare the costPrime for the previous layer for cost error back propagation
            double[][] weightTranspose = MatrixManipulator.transposeMatrix(weight);
            costPrime = MatrixManipulator.multiply(costOutputPrime, weightTranspose);
//            MatrixManipulator.debugMatrix(layerName + " costPrime: ", costPrime);

            // eliminate the first col as it is for bias
            costPrime = MatrixManipulator.removeFirstColumn(costPrime);
//            MatrixManipulator.debugMatrix(layerName + " costPrime [*First column removed]:", costPrime);
        }

        logger.debug("Back-propagate to the next layer");
        previousLayer.backPropagation(costPrime);
    }

    private void accumulateDeltaWeights(double[][] dWeights) {
        assert (deltaWeight.length == dWeights.length);
        assert (deltaWeight[0].length == dWeights[0].length);

//        MatrixManipulator.debugMatrix("accumulateDeltaWeights::dWeights", dWeights);

        for (int i = 0; i < deltaWeight.length; i++) {
            for (int j = 0; j < deltaWeight[0].length; j++) {
                deltaWeight[i][j] += dWeights[i][j];
            }
        }

//        MatrixManipulator.debugMatrix("accumulateDeltaWeights", deltaWeight);
    }

    public Layer input(double[][] input) {
        this.input = input;
//        MatrixManipulator.debugMatrix(layerName + " received input: ", input);

        return this;
    }

    public void setPreviousLayer(Layer layer) {
        previousLayer = layer;
    }

    public void setNextLayer(Layer layer) {
        nextLayer = layer;
    }

    public abstract double[][] forwardPropagation();

    public abstract void backPropagation(double[][] costOutputPrime);

    public abstract int getNumberOfOutputs();

    public abstract ActivationFunction getActivationFunction();

    public void updateWeights() {

//***********************************************************
//        if ((this instanceof HiddenLayer)) {
//            MatrixManipulator.debugMatrix("Hidden Layer Weight [Before]", weight);
//        }
//
//        if ((this instanceof OutputLayer)) {
//            MatrixManipulator.debugMatrix("Output Layer Weight [Before]", weight);
//        }
//***********************************************************

        // not applicable for Input Layer
        if (!(this instanceof InputLayer)) {
            assert (weight.length == deltaWeight.length);
            assert (weight[0].length == deltaWeight[0].length);

            logger.debug("{} update weights", layerName);
            for (int i = 0; i < weight.length; i++) {
                for (int j = 0; j < weight[0].length; j++) {
                    weight[i][j] -= learningRate * deltaWeight[i][j];
                }
            }

//            MatrixManipulator.debugMatrix("deltas", deltaWeight);
        }

//***********************************************************
//        if ((this instanceof HiddenLayer)) {
//            MatrixManipulator.debugMatrix("Hidden Layer Weight [After]", weight);
//        }
//
//        if ((this instanceof OutputLayer)) {
//            MatrixManipulator.debugMatrix("Output Layer Weight [After]", weight);
//        }
//***********************************************************

        if (null != nextLayer) {
            logger.debug("Update weights for the next Layer");
            nextLayer.updateWeights();
        }
    }

    public double[][] getW() {
        return weight;
    }

    public double[][] getA() {
        return A;
    }

    public double[][] getZ() {
        return Z;
    }
}
