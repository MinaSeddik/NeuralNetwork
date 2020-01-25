package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MaxPooling2D extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(MaxPooling2D.class);

    private Quartet<Integer, Integer, Integer, Integer> inputShape;
    private Pair<Integer, Integer> poolSize;

    public MaxPooling2D(Pair<Integer, Integer> poolSize) {
        this.poolSize = poolSize;
    }

    @Override
    public void buildupLayer() {

        // do nothing for now

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
        this.inputShape = (Quartet<Integer, Integer, Integer, Integer>)inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Quartet<>(inputShape.getValue0(), inputShape.getValue1(), inputShape.getValue2(), inputShape.getValue3());
    }
}
