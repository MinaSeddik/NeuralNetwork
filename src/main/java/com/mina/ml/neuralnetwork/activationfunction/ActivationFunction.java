package com.mina.ml.neuralnetwork.activationfunction;

import com.mina.ml.neuralnetwork.util.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class ActivationFunction implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(ActivationFunction.class);

    private final static int NUM_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();

    private static ExecutorService executor = Executors.newFixedThreadPool(NUM_OF_PROCESSORS);

    public void activate(double[][] matrix, double[][] result, int startIndex, int endIndex) {

        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activate(matrix[i][j]);
            }
        }
    }

    public double[][] activate(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        if (matrix.length <= NUM_OF_PROCESSORS) {
            activate(matrix, result, 0, matrix.length);
        } else {
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
        }

        return result;
    }

    /* new implementation */
    public Matrix activate(Matrix matrix) {
        return matrix.apply(val -> activate(val));
    }

    public Matrix activatePrime(Matrix matrix) {
        return matrix.apply(val -> activatePrime(val));
    }

    public void activatePrime(double[][] matrix, double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = activatePrime(matrix[i][j]);
            }
        }
    }


    public double[][] activatePrime(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        if (matrix.length <= NUM_OF_PROCESSORS) {
            activatePrime(matrix, result, 0, matrix.length);
        } else {
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
        }

        return result;
    }

    public abstract double activate(double value);

    public abstract double activatePrime(double value);
}
