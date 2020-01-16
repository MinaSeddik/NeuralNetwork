package com.mina.ml.neuralnetwork.factory;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.layer.Layerrr;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
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

    public void optimize(Layerrr inputLayer, Layerrr outputLayer, LossFunction lossFunction, Matrix x, Matrix y){

        Matrix yPrime = inputLayer.forwardPropagation(x);

        double meanError = lossFunction.meanErrorCost(y, yPrime);

        Matrix errorCostPrime = lossFunction.errorCostPrime(y, yPrime);

        outputLayer.backPropagation(errorCostPrime);

//        inputLayer.updateWeights();


    }

}
