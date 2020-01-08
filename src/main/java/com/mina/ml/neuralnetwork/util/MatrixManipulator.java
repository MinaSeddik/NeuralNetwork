package com.mina.ml.neuralnetwork.util;

import com.google.common.base.Stopwatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by menai on 2019-02-01.
 */
public class MatrixManipulator {

    private final static Logger logger = LoggerFactory.getLogger(MatrixManipulator.class);

    private final static int NUM_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();

    private static ExecutorService executor = Executors.newFixedThreadPool(NUM_OF_PROCESSORS);
    private static ExecutorService executor2 = Executors.newFixedThreadPool(NUM_OF_PROCESSORS);

    public static float[][] addColumnOfOnes(float[][] input) {
        float[][] matrix = new float[input.length][input[0].length + 1];

        for (int row = 0; row < input.length; row++) {
            matrix[row][0] = 1f;
            for (int i = 1, j = 0; i < matrix[0].length && j < input[0].length; i++, j++) {
                matrix[row][i] = input[row][j];
            }
        }
        return matrix;
    }

    public static float[][] multiply_singleThread(float[][] matrix1, float[][] matrix2) {
        if (matrix1[0].length != matrix2.length) {
            throw new RuntimeException("Can't Multiply Matrices of different Dimensions");
        }

        float[][] result = new float[matrix1.length][matrix2[0].length];
        initializeMatrix(result, 0f);

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix2.length; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    public static void multiply(float[][] matrix1, float[][] matrix2, float[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix2.length; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    public static float[][] multiply(float[][] matrix1, float[][] matrix2) {
        if (matrix1[0].length != matrix2.length) {
            throw new RuntimeException("Can't Multiply Matrices of different Dimensions");
        }
        float[][] result = new float[matrix1.length][matrix2[0].length];
//        initializeMatrix(result, 0f);
//        System.out.println("start ... ");
//        Stopwatch stopwatch = Stopwatch.createStarted();

        List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                .mapToObj(p -> {
                    Future<?> future = executor.submit(() -> multiply(matrix1, matrix2, result,
                            p * (matrix1.length / NUM_OF_PROCESSORS),
                            p == NUM_OF_PROCESSORS ?
                                    matrix1.length :
                                    p * (matrix1.length / NUM_OF_PROCESSORS) + (matrix1.length / NUM_OF_PROCESSORS)));
                    return future;
                }).collect(Collectors.toList());

        futures.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException("Exception: " + ex.getClass() + " " + ex.getMessage());
            }
        });
//        long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
//        System.out.println("end ... Elapsed time = " + timeElapsed);
        return result;
    }

    public static void initializeMatrix(float[][] matrix, float val) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = val;
            }
        }
    }

    public static float[][] transposeMatrix(float[][] matrix) {
        float[][] result = new float[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    public static float[][] vectorToMatrix(float[] vector) {
        float[][] result = new float[1][vector.length];

        for (int i = 0; i < vector.length; i++) {
            result[0][i] = vector[i];
        }

        return result;
    }

    public static float[][] removeFirstColumn(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length - 1];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                result[i][j - 1] = matrix[i][j];
            }
        }

        return result;
    }

    public static float[][] multiplyEntries_singleThread(float[][] matrix1, float[][] matrix2) {
        assert (matrix1.length == matrix2.length);
        assert (matrix1[0].length == matrix2[0].length);

        float[][] result = new float[matrix1.length][matrix1[0].length];

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        return result;
    }

    public static float[][] multiplyEntries(float[][] matrix1, float[][] matrix2, float[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        return result;
    }

    public static float[][] multiplyEntries(float[][] matrix1, float[][] matrix2) {
        assert (matrix1.length == matrix2.length);
        assert (matrix1[0].length == matrix2[0].length);

        float[][] result = new float[matrix1.length][matrix1[0].length];

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                .mapToObj(p -> {
                    Future<?> future = executor2.submit(() -> multiplyEntries(matrix1, matrix2, result,
                            p * (matrix1.length / NUM_OF_PROCESSORS),
                            p == NUM_OF_PROCESSORS ?
                                    matrix1.length :
                                    p * (matrix1.length / NUM_OF_PROCESSORS) + (matrix1.length / NUM_OF_PROCESSORS)));
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

    public static void debugMatrix(String label, float[][] matrix) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < matrix.length; x++) {
            for (float y : matrix[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < matrix.length - 1 ? "\n" : "");
        }
        logger.debug(matrixAsString.toString());
    }

    public static void printMatrix(String label, float[][] matrix) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < matrix.length; x++) {
            for (float y : matrix[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < matrix.length - 1 ? "\n" : "");
        }
        logger.info(matrixAsString.toString());
    }
}
