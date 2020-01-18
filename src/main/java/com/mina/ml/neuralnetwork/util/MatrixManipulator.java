package com.mina.ml.neuralnetwork.util;

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
 * Created by menai on 2019-02-01.
 */
public class MatrixManipulator {

    private final static Logger logger = LoggerFactory.getLogger(MatrixManipulator.class);

    private final static int NUM_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();

    private static ExecutorService executor = Executors.newFixedThreadPool(NUM_OF_PROCESSORS);

    public static void addColumnOfOnes(double[][] input, double[][] result, int startIndex, int endIndex) {

//        System.out.println("add onces start = " + startIndex + ", end = " + endIndex);
        for (int row = startIndex; row < endIndex; row++) {
            result[row][0] = 1d;
            for (int i = 1, j = 0; i < result[0].length && j < input[0].length; i++, j++) {
                result[row][i] = input[row][j];
            }
        }

    }

    public static double[][] addColumnOfOnes(double[][] input) {
        double[][] result = new double[input.length][input[0].length + 1];

        if (input.length <= NUM_OF_PROCESSORS) {
            addColumnOfOnes(input, result, 0, input.length);
        } else {


            List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                    .mapToObj(p -> {
                        Future<?> future = executor.submit(() -> addColumnOfOnes(input, result,
                                p * (input.length / NUM_OF_PROCESSORS),
                                p == NUM_OF_PROCESSORS ?
                                        input.length :
                                        p * (input.length / NUM_OF_PROCESSORS) + (input.length / NUM_OF_PROCESSORS)));
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

    public static double[][] multiply_singleThread(double[][] matrix1, double[][] matrix2) {
        if (matrix1[0].length != matrix2.length) {
            throw new RuntimeException("Can't Multiply Matrices of different Dimensions");
        }

        double[][] result = new double[matrix1.length][matrix2[0].length];
        initializeMatrix(result, 0d);

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix2.length; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    public static void multiply(double[][] matrix1, double[][] matrix2, double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix2[0].length; j++) {
                for (int k = 0; k < matrix2.length; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
    }

    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        if (matrix1[0].length != matrix2.length) {
            throw new RuntimeException("Can't Multiply Matrices of different Dimensions");
        }
        double[][] result = new double[matrix1.length][matrix2[0].length];
//        initializeMatrix(result, 0f);
//        System.out.println("start ... ");
//        Stopwatch stopwatch = Stopwatch.createStarted();

        if (matrix1.length <= NUM_OF_PROCESSORS) {
            multiply(matrix1, matrix2, result, 0, matrix1.length);
        } else {
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
        }
//        long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
//        System.out.println("end ... Elapsed time = " + timeElapsed);
        return result;
    }

    public static void initializeMatrix(double[][] matrix, double val) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = val;
            }
        }
    }

    public static void transposeMatrix(double[][] matrix, double[][] result, int startIndex, int endIndex) {

        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }

    }


    public static double[][] transposeMatrix(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];

        if (matrix.length <= NUM_OF_PROCESSORS) {
            transposeMatrix(matrix, result, 0, matrix.length);
        } else {
            List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                    .mapToObj(p -> {
                        Future<?> future = executor.submit(() -> transposeMatrix(matrix, result,
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

    public static double[][] vectorToMatrix(double[] vector) {
        double[][] result = new double[1][vector.length];

        for (int i = 0; i < vector.length; i++) {
            result[0][i] = vector[i];
        }

        return result;
    }

    public static void removeFirstColumn(double[][] matrix, double[][] result, int startIndex, int endIndex) {

        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                result[i][j - 1] = matrix[i][j];
            }
        }
    }


    public static double[][] removeFirstColumn(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length - 1];

        if (matrix.length <= NUM_OF_PROCESSORS) {
            removeFirstColumn(matrix, result, 0, matrix.length);
        } else {
            List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                    .mapToObj(p -> {
                        Future<?> future = executor.submit(() -> removeFirstColumn(matrix, result,
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


    public static double[][] multiplyEntries_singleThread(double[][] matrix1, double[][] matrix2) {
        assert (matrix1.length == matrix2.length);
        assert (matrix1[0].length == matrix2[0].length);

        double[][] result = new double[matrix1.length][matrix1[0].length];

        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        return result;
    }

    public static double[][] multiplyEntries(double[][] matrix1, double[][] matrix2, double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < matrix1[0].length; j++) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        return result;
    }

    public static double[][] multiplyEntries(double[][] matrix1, double[][] matrix2) {
        assert (matrix1.length == matrix2.length);
        assert (matrix1[0].length == matrix2[0].length);

        double[][] result = new double[matrix1.length][matrix1[0].length];

//        for (int i = 0; i < matrix1.length; i++) {
//            for (int j = 0; j < matrix1[0].length; j++) {
//                result[i][j] = matrix1[i][j] * matrix2[i][j];
//            }
//        }

        if (matrix1.length <= NUM_OF_PROCESSORS) {
            multiplyEntries(matrix1, matrix2, result, 0, matrix1.length);
        } else {
            List<Future<?>> futures = IntStream.range(0, NUM_OF_PROCESSORS)
                    .mapToObj(p -> {
                        Future<?> future = executor.submit(() -> multiplyEntries(matrix1, matrix2, result,
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
        }

        return result;
    }

    public static void debugMatrix2(String label, double[][] matrix) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < matrix.length; x++) {
            for (double y : matrix[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < matrix.length - 1 ? "\n" : "");
        }
//        logger.debug(matrixAsString.toString());
        System.out.println(matrixAsString);
    }

    public static void printMatrix(String label, double[][] matrix) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < matrix.length; x++) {
            for (double y : matrix[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < matrix.length - 1 ? "\n" : "");
        }
        logger.info(matrixAsString.toString());
    }

    public static String simulate(String label, int rows, int cols, char r, char c) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        boolean rowPrinted;
        boolean colPrinted;
        int voidRowsCount = 0;
        String val;
        for (int x = 0; x < rows; x++) {
            colPrinted = false;
            rowPrinted = false;
            for (int y = 0; y < cols; y++) {
                if ((x == 0 || x == 1 || x == 2 || x + 1 == rows || x + 2 == rows) &&
                        (y <= 3 || y + 1 == cols || y + 2 == cols)) {
                    val = String.format("%c%d%c%d\t", r, x, c, y);
                    rowPrinted = true;
                } else if (rowPrinted && !colPrinted) {
                    val = "...  ";
                    colPrinted = true;
                } else if (!rowPrinted && !colPrinted && voidRowsCount < 3) {
                    val = "..\t..";
                    rowPrinted = true;
                    colPrinted = true;
                    voidRowsCount++;
                } else {
                    val = "";
                }
                matrixAsString.append(val);
            }
            matrixAsString.append(rowPrinted || colPrinted || voidRowsCount < 3 ? "\n":"");
        }
//        logger.info(matrixAsString.toString());
        return matrixAsString.toString();
    }
}
