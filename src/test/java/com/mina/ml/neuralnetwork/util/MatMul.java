package com.mina.ml.neuralnetwork.util;

import org.junit.Test;

import java.util.Random;

public class MatMul {

    private final static int ROWS1 = 100;
    private final static int COLS1 = 150;

    private final static int ROWS2 = 150;
    private final static int COLS2 = 200;

    @Test
    public void matmul() {
        double[][] matrix1 = new double[ROWS1][COLS1];
        double[][] matrix2 = new double[ROWS2][COLS2];

        generateRandom(matrix1);
        generateRandom(matrix2);

        double[][] r1 = MatrixManipulator.multiply_singleThread(matrix1, matrix2);
        double[][] r2 = MatrixManipulator.multiply(matrix1, matrix2);

        assertMatrices(r1, r2);

        Matrix m1 = new Matrix(matrix1);
        Matrix m2 = new Matrix(matrix2);
        double[][] r3 = m1.dot(m2).getMatrix();

        assertMatrices(r1, r3);

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
