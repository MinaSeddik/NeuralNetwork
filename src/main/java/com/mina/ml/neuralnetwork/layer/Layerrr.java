package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public abstract class Layerrr implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Layerrr.class);



    protected int layerIndex;

    protected Layerrr prevLayer;
    protected Layerrr nextLayer;

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
        return prevLayer;
    }

    public void setPrevious(Layerrr layer) {
        prevLayer = layer;
    }

    public void setNext(Layerrr layer) {
        nextLayer = layer;
        networkLayerType = NetworkLayerType.HIDDEN;
    }

    public String getLayerType() {
        return this.getClass().getSimpleName();
    }

    public abstract void buildupLayer();

    public abstract String getName();

    public abstract int getNumberOfParameter();

    public abstract Tensor forwardPropagation(Tensor inputTensor);

    public abstract void printForwardPropagation(Tensor input);

    public abstract void backPropagation(Tensor costPrime);

    public abstract void updateWeight(double learningRate);

    public abstract Tensor getWeights();

    public abstract void setWeights(Tensor weight);

    public abstract void setInputShape(Tuple inputShape);

    public abstract Tuple getOutputShape();

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

}
