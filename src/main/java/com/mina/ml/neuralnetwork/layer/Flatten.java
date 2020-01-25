package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Flatten extends Layerrr{

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Flatten.class);

    private Tuple inputShape;

    @Override
    public void buildupLayer() {

    }

    @Override
    public String getName() {
        return "xxxxx_" + layerIndex;
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
        this.inputShape = inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Pair<Integer, Integer>(0, 1000000000);
    }
}
