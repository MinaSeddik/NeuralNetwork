package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Relu extends ActivationFunction {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Relu.class);

    @Override
    public double activate(double value) {
        return Math.max(0d, value);
    }

    @Override
    public double activatePrime(double value) {
        //https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
        assert (value != 0);

//        System.out.println(value);
        // if x == 0 then the Relu is UNDEFINED
        return value <= 0d ? 0d : 1d;
    }

}
