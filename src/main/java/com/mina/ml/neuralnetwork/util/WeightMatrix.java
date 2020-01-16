package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class WeightMatrix extends Matrix {

    private final static Logger logger = LoggerFactory.getLogger(WeightMatrix.class);

    public WeightMatrix(int rows, int columns) {
        super(rows, columns);
    }

    public WeightMatrix(double[][] matrix) {
        super(matrix);
    }

    public Matrix updateWeights(Matrix deltaWeight, double learningRate){
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> updateWeights(result, deltaWeight.collection, learningRate, start, end));
        collection = result;

        return this;
    }

    public Matrix initializeRandom(double min, double max) {
        Random random = new Random();
        logger.debug(String.format("Initializing Weights between [%.2f] and [%.2f]", min, max));
        parallelizeOperation((start, end) -> initializeRandom(random, min, max, start, end));

        return this;
    }

    private void updateWeights(double[][] result, double[][] deltaWeight, double learningRate, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] -= learningRate * deltaWeight[i][j];
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
