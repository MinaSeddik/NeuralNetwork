package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Tansh extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(Tansh.class);

    @Override
    public double activate(double value) {
        // (e^x - e^-x) / (e^x + e^-x)
        return ((Math.exp((double) value) - Math.exp((double) -value)) /
                (Math.exp((double) value) + Math.exp((double) -value)));
    }

    @Override
    public double activatePrime(double value) {
        return 1d - Math.pow(activate(value), 2);
    }

}
