package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VoidActivation extends ActivationFunction {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(VoidActivation.class);

    @Override
    public double activate(double value) {
        return value;
    }

    @Override
    public double activatePrime(double value) {
        return 1d;
    }

}
