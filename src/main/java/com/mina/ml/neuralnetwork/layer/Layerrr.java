package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Matrix;

public abstract class Layerrr {

    protected int layerIndex;

    protected Layerrr previousDense;
    protected Layerrr nextDense;

    protected int numOfInputs;
    protected int numOfOutputs;

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

    public abstract void backPropagation(Matrix costPrime);

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
