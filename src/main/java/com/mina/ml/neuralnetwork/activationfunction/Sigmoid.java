package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Sigmoid extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(Sigmoid.class);

    @Override
    public float activate(float value) {
        //f(x)=1/(1+e^-x)
        return 1f / (1f + (float) Math.exp(-value));
    }

    @Override
    public float activatePrime(float value) {
        return activate(value) * (1 - activate(value));
    }

}
