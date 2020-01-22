package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Sigmoid extends ActivationFunction {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Sigmoid.class);

    @Override
    public double activate(double value) {
        //f(x)=1/(1+e^-x)
        return 1d / (1d + Math.exp(-value));
    }

    @Override
    public double activatePrime(double value) {
        return activate(value) * (1 - activate(value));
    }

}
