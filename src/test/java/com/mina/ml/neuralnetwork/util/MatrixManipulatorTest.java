package com.mina.ml.neuralnetwork.util;


import com.google.common.base.Stopwatch;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.TimeUnit;

public class MatrixManipulatorTest {

    private final static int ROWS1 = 2000;
    private final static int COLS1 = 1500;

    private final static int ROWS2 = 1500;
    private final static int COLS2 = 1000;

    @Test
    public void multiply() {
        double[][] matrix1 = new double[ROWS1][COLS1];
        double[][] matrix2 = new double[ROWS2][COLS2];


        System.out.println("Generating Randoms for M1");
        generateRandom(matrix1);

        System.out.println("Generating Randoms for M2");
        generateRandom(matrix2);

        System.out.println("Mul Single Thread");
        Stopwatch stopwatch = Stopwatch.createStarted();
        double[][] result1 = MatrixManipulator.multiply_singleThread(matrix1, matrix2);
        stopwatch.stop();
        long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.println("Elapsed time = " + timeElapsed);

        System.out.println("Mul Multi-Thread");
        stopwatch = Stopwatch.createStarted();
        double[][] result2 = MatrixManipulator.multiply(matrix1, matrix2);
        stopwatch.stop();
        timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.println("Elapsed time = " + timeElapsed);

        assertMatrices(result1, result2);
    }

    @Test
    public void multiplyEntries() {
        double[][] matrix1 = new double[ROWS1][COLS1];
        double[][] matrix2 = new double[ROWS1][COLS1];

        System.out.println("Generating Randoms for M1");
        generateRandom(matrix1);

        System.out.println("Generating Randoms for M2");
        generateRandom(matrix2);

        System.out.println("Mul-Entries Single Thread");
        Stopwatch stopwatch = Stopwatch.createStarted();
        double[][] result1 = MatrixManipulator.multiplyEntries_singleThread(matrix1, matrix2);
        stopwatch.stop();
        long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.println("Elapsed time = " + timeElapsed);

        System.out.println("Mul-Entries Multi-Thread");
        stopwatch = Stopwatch.createStarted();
        double[][] result2 = MatrixManipulator.multiplyEntries(matrix1, matrix2);
        stopwatch.stop();
        timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.println("Elapsed time = " + timeElapsed);

        assertMatrices(result1, result2);
    }

    private void assertMatrices(double[][] result1, double[][] result2) {

        for (int i = 0; i < result1.length; i++) {
            for (int j = 0; j < result1[0].length; j++) {
                assert result1[i][j] == result2[i][j];
            }
        }
    }

    private void generateRandom(double[][] matrix) {
        Random r = new Random();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = r.nextDouble();
            }
        }
    }
}