package com.mina.ml.neuralnetwork.layer;


// https://wiseodd.github.io/techblog/2016/06/25/dropout/

import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Dropout extends Layer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Dropout.class);

    private Tuple inputShape;
    private float dropoutProbability;

    public Dropout(float dropoutProbability) {
        this.dropoutProbability = dropoutProbability;
    }

    @Override
    public void buildupLayer() {

    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public int getNumberOfParameter() {
        return 0;
    }

    @Override
    public Tensor forwardPropagation(Tensor input) {
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
        // Non-Applicable for this Layer
    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public void setWeights(Tensor weight) {
        // Non-Applicable for this Layer
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return inputShape;
    }
}
