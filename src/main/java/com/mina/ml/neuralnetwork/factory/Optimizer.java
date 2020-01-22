package com.mina.ml.neuralnetwork.factory;

import com.mina.ml.neuralnetwork.layer.Layerrr;
import com.mina.ml.neuralnetwork.layer.Verbosity;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

public class Optimizer implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Optimizer.class);

    private final static double LEARNING_RATE = 0.001;

    private double learningRate;

    private Verbosity verbosity = Verbosity.NORMAL;

    public Optimizer() {
        this(LEARNING_RATE);
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setVerbosity(Verbosity verbosity) {
        this.verbosity = verbosity;
    }

    public Pair<Double, Double> optimize(Layerrr inputLayer, Layerrr outputLayer, LossFunction lossFunction, Matrix x, Matrix y) {

        Matrix yPrime = inputLayer.forwardPropagation(x);

        double loss = lossFunction.meanErrorCost(y, yPrime);
        double acc = lossFunction.calculateAccuracy(y, yPrime);

        Matrix errorCostPrime = lossFunction.errorCostPrime(y, yPrime);

        outputLayer.backPropagation(errorCostPrime);

        inputLayer.updateWeight(learningRate);

        return new Pair<>(loss, acc);
    }

}
