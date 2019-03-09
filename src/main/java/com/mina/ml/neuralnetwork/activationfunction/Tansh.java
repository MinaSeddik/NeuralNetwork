package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public class Tansh extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(Tansh.class);

    @Override
    public float activate(float value) {
        // (e^x - e^-x) / (e^x + e^-x)
        return (float) ((Math.exp((double) value) - Math.exp((double) -value)) /
                (Math.exp((double) value) + Math.exp((double) -value)));
    }

    @Override
    public float activatePrime(float value) {
        return 1f - (float) Math.pow(activate(value), 2);
    }

}
