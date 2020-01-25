package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class D4WeightMatrix extends D4Matrix {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(D4WeightMatrix.class);

    public D4WeightMatrix(int dimension, int depth, int rows, int columns) {
        super(dimension, depth, rows, columns);
    }

    public D4WeightMatrix(double[][][][] d4Matrix) {
        super(d4Matrix);
    }

    public D4WeightMatrix initializeRandom(double min, double max) {

        Random random = new Random();
        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", min, max));
        parallelizeOperation((start, end) -> initializeRandom(random, min, max, start, end));

        return this;
    }

    private void initializeRandom(Random random, double minRange, double maxRange, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                for (int k = 0; k < collection[i][j].length; k++) {
                    for (int l = 0; l < collection[i][j][k].length; l++) {
                        collection[i][j][k][l] = minRange + (maxRange - minRange) * random.nextDouble();
                    }
                }
            }
        }
    }
}
