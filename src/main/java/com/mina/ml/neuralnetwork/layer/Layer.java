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

    protected float[][] input;
    protected float[][] A;
    protected float[][] Z;

    protected float[][] weight;
    protected float[][] deltaWeight;

    protected ActivationFunction activationFunction;
    protected double learningRate;

    public Layer(String layerName, int numOfInputs, int numOfOutputs, String activationFunctionName, double learningRate) {
        this.layerName = layerName;

        this.numOfInputs = numOfInputs;
        this.numOfOutputs = numOfOutputs;

        this.learningRate = learningRate;

        // init the weight matrix
        weight = new float[numOfInputs][numOfOutputs];
        initializeWeights(weight);

        deltaWeight = new float[numOfInputs][numOfOutputs];

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

    private void initializeWeights(float[][] weight) {
        float rangeMin = -1.0f;
        float rangeMax = 1.0f;
        Random r = new Random();

        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", rangeMin, rangeMax));
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[0].length; j++) {
                weight[i][j] = rangeMin + (rangeMax - rangeMin) * r.nextFloat();
            }
        }
    }

    protected void calculateDeltaWeight(float[][] costOutputPrime) {
        // make sure that both of them has the same row ( # of examples)
        assert (costOutputPrime.length == input.length);

        assert (deltaWeight.length == input[0].length);
        assert (deltaWeight[0].length == costOutputPrime.length);

        // reset delta weights
        MatrixManipulator.initializeMatrix(deltaWeight, 0f);

        for (int i = 0; i < input.length; i++) {
//            logger.info("HERE 7.3.1");
            float[][] in = MatrixManipulator.transposeMatrix(MatrixManipulator.vectorToMatrix(input[i]));
            float[][] out = MatrixManipulator.vectorToMatrix(costOutputPrime[i]);
//            logger.info("HERE 7.3.2");
//            MatrixManipulator.debugMatrix(layerName + " in: ", in);
//            MatrixManipulator.debugMatrix(layerName + " out: ", out);
//            logger.info("HERE 7.3.3");
            float[][] dWeights = MatrixManipulator.multiply(in, out);
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

    protected void prepareErrorCostThenBackPropagate(float[][] costOutputPrime) {
        float[][] costPrime = null;

        if (previousLayer instanceof HiddenLayer) {

            // prepare the costPrime for the previous layer for cost error back propagation
            float[][] weightTranspose = MatrixManipulator.transposeMatrix(weight);
            costPrime = MatrixManipulator.multiply(costOutputPrime, weightTranspose);
//            MatrixManipulator.debugMatrix(layerName + " costPrime: ", costPrime);

            // eliminate the first col as it is for bias
            costPrime = MatrixManipulator.removeFirstColumn(costPrime);
//            MatrixManipulator.debugMatrix(layerName + " costPrime [*First column removed]:", costPrime);
        }

        logger.debug("Back-propagate to the next layer");
        previousLayer.backPropagation(costPrime);
    }

    private void accumulateDeltaWeights(float[][] dWeights) {
        assert (deltaWeight.length == dWeights.length);
        assert (deltaWeight[0].length == dWeights[0].length);

        for (int i = 0; i < deltaWeight.length; i++) {
            for (int j = 0; j < deltaWeight[0].length; j++) {
                deltaWeight[i][j] += dWeights[i][j];
            }
        }
    }

    public Layer input(float[][] input) {
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

    public abstract float[][] forwardPropagation();

    public abstract void backPropagation(float[][] costOutputPrime);

    public abstract int getNumberOfOutputs();

    public abstract ActivationFunction getActivationFunction();

    public void updateWeights() {

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

        }

        if (null != nextLayer) {
            logger.debug("Update weights for the next Layer");
            nextLayer.updateWeights();
        }
    }
}
