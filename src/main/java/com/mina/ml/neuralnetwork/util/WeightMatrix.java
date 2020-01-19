package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.function.Function;

public class WeightMatrix extends Matrix {

    private final static Logger logger = LoggerFactory.getLogger(WeightMatrix.class);

    public WeightMatrix(int rows, int columns) {
        super(rows, columns);
    }

    public WeightMatrix(double[][] matrix) {
        super(matrix);
    }

    public Matrix updateWeights(Matrix deltaWeight, double learningRate){
        parallelizeOperation((start, end) -> updateWeights(deltaWeight.collection, learningRate, start, end));

        return this;
    }

    public Matrix initializeRandom(double min, double max) {
//        double rangeMin = -1.0d;
//        double rangeMax = 1.0d;
        Random random = new Random();
//        r.setSeed(100);


//        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", rangeMin, rangeMax));
//        for (int i = 0; i < collection.length; i++) {
//            for (int j = 0; j < collection[0].length; j++) {
//                collection[i][j] = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
//            }
//        }
        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", min, max));
        parallelizeOperation((start, end) -> initializeRandom(random, min, max, start, end));

        return this;
    }

    private void updateWeights(double[][] deltaWeight, double learningRate, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] -= learningRate * deltaWeight[i][j];
            }
        }
    }

    private void initializeRandom(Random random, double minRange, double maxRange, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] = minRange + (maxRange - minRange) * random.nextDouble();
            }
        }
    }
}
