package com.mina.ml.neuralnetwork.activationfunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(ActivationFunction.class);

    private final static int NUM_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();

    private static ExecutorService executor = Executors.newFixedThreadPool(NUM_OF_PROCESSORS);

    public void activate(float[][] matrix, float[][] result, int startIndex, int endIndex) {

        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activate(matrix[i][j]);
            }
        }
    }

    public float[][] activate(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                .mapToObj(p -> {
                    Future<?> future = executor.submit(() -> activate(matrix, result,
                            p * (matrix.length / NUM_OF_PROCESSORS),
                            p == NUM_OF_PROCESSORS ?
                                    matrix.length :
                                    p * (matrix.length / NUM_OF_PROCESSORS) + (matrix.length / NUM_OF_PROCESSORS)));
                    return future;
                }).collect(Collectors.toList());

        futures.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException("Exception: " + ex.getClass() + " " + ex.getMessage());
            }
        });

        return result;
    }

    public void activatePrime(float[][] matrix, float[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activatePrime(matrix[i][j]);
            }
        }
    }


    public float[][] activatePrime(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                .mapToObj(p -> {
                    Future<?> future = executor.submit(() -> activatePrime(matrix, result,
                            p * (matrix.length / NUM_OF_PROCESSORS),
                            p == NUM_OF_PROCESSORS ?
                                    matrix.length :
                                    p * (matrix.length / NUM_OF_PROCESSORS) + (matrix.length / NUM_OF_PROCESSORS)));
                    return future;
                }).collect(Collectors.toList());

        futures.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException("Exception: " + ex.getClass() + " " + ex.getMessage());
            }
        });

        return result;
    }

    public abstract float activate(float value);

    public abstract float activatePrime(float value);
}
