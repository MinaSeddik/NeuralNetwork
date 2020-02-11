package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class BiasVector extends Vector {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(BiasVector.class);

    public BiasVector(int count) {
        super(count);
    }

    public BiasVector(double[] list) {
        super(list);
    }

    public void updateBias(Vector deltaBias, double learningRate) {
        for (int i = 0; i < collection.length; i++) {
            collection[i] -= learningRate * deltaBias.collection[i];
        }
    }

    public BiasVector initializeRandom(double min, double max) {
        Random random = new Random();

        logger.debug(String.format("Initializing Bias between [%.2f] and [%.2f]", min, max));
        parallelizeOperation((start, end) -> initializeRandom(random, min, max, start, end));

        return this;
    }

    private void initializeRandom(Random random, double minRange, double maxRange, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            collection[i] = minRange + (maxRange - minRange) * random.nextDouble();
        }
    }

}
