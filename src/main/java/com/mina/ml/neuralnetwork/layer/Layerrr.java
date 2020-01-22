package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.WeightMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public abstract class Layerrr implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Layerrr.class);

    protected Matrix input;
    protected Matrix A;
    protected Matrix Z;

    protected int layerIndex;

    protected Layerrr previousDense;
    protected Layerrr nextDense;

    protected int numOfInputs;
    protected int numOfOutputs;

    protected WeightMatrix weight;

    protected NetworkLayerType networkLayerType = NetworkLayerType.OUTPUT;

    protected ActivationFunction activationFunction;

    public void setIndex(int index) {
        layerIndex = index;
    }

    public int getIndex() {
        return layerIndex;
    }

    public Layerrr getPrev() {
        return previousDense;
    }

    public void setPreviousDense(Layerrr layer) {
        previousDense = layer;
    }

    public void setNextDense(Layerrr layer) {
        nextDense = layer;
        networkLayerType = NetworkLayerType.HIDDEN;
    }

    public String getType() {
        return this.getClass().getSimpleName();
    }

    public int getOutputParameters() {
        return numOfOutputs;
    }

    public void setInputParameters(int paramCount) {
        /* Add 1 for Bias */
        numOfInputs = paramCount + 1;
    }

    public abstract String getName();

    public abstract int getNumberOfParameter();

    public abstract Matrix forwardPropagation(Matrix input);

    public abstract void printForwardPropagation(Matrix input);

    public abstract void backPropagation(Matrix costPrime);

    public abstract void updateWeight(double learningRate);

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public WeightMatrix getWeights() {
        return weight;
    }

    public void setWeights(WeightMatrix weight) {
        this.weight = weight;
    }

}
