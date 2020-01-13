package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Relu extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(Relu.class);

    @Override
    public double activate(double value) {
        return Math.max(0f, value);
    }

    @Override
    public double activatePrime(double value) {
        //https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
        assert (value != 0);

        // if x == 0 then the Relu is UNDEFINED
        return value < 0f ? 0f : 1f;
    }

}
