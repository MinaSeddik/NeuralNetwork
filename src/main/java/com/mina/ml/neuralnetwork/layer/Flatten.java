package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Triplet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Flatten extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Flatten.class);

    private Tuple inputShape;

    @Override
    public void buildupLayer() {

    }

    @Override
    public String getName() {
        return "flatten_" + layerIndex;
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
        int links = 0;
        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Pair":
                links = ((Pair<Integer, Integer>) inputShape).getValue0();
                break;
            case "org.javatuples.Triplet":
                Triplet<Integer, Integer, Integer> shape3 = (Triplet<Integer, Integer, Integer>) inputShape;
                links = shape3.getValue1() * shape3.getValue2();
                break;
            case "org.javatuples.Quartet":
                Quartet<Integer, Integer, Integer, Integer> shape4 = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
                links = shape4.getValue1() * shape4.getValue2() * shape4.getValue3();
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
        return new Pair<Integer, Integer>(0, links);
    }
}
