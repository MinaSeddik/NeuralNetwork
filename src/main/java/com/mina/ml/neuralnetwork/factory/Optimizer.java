package com.mina.ml.neuralnetwork.factory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Optimizer {

    private final static Logger logger = LoggerFactory.getLogger(Optimizer.class);

    private final static double LEARNING_RATE = 0.001;

    private double learningRate;

    public Optimizer() {
        this(LEARNING_RATE);
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }


}
