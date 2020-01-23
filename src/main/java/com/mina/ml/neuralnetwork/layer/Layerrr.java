package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.Tensor;
import com.mina.ml.neuralnetwork.util.WeightMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public abstract class Layerrr implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Layerrr.class);



    protected int layerIndex;

    protected Layerrr previousDense;
    protected Layerrr nextDense;

    @Deprecated
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

    public void setPrevious(Layerrr layer) {
        previousDense = layer;
    }

    public void setNext(Layerrr layer) {
        nextDense = layer;
        networkLayerType = NetworkLayerType.HIDDEN;
    }

    public String getLayerType() {
        return this.getClass().getSimpleName();
    }

    public abstract void buildupLayer();

    public abstract String getName();

    public abstract int getNumberOfParameter();

    public abstract Matrix forwardPropagation(Matrix input);

    public abstract void printForwardPropagation(Matrix input);

    public abstract void backPropagation(Matrix costPrime);

    public abstract void updateWeight(double learningRate);

    public abstract Tensor getWeights();

    public abstract void setWeights(Tensor weight);

    // I should re-visit it
    public abstract void setInputParameters(int paramCount);

    // I should re-visit it
    public abstract int getOutputParameters();

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

}
