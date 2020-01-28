package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class D4Matrix extends Tensor {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

    protected double[][][][] collection;

    public D4Matrix(int dimension, int depth, int rows, int columns) {
        collection = new double[dimension][depth][rows][columns];
    }

    public D4Matrix(double[][][][] d4Matrix) {
        collection = d4Matrix;
    }

    public D4Matrix(List<double[][][]> list) {
        collection = new double[list.size()][list.get(0).length][list.get(0)[0].length][list.get(0)[0][0].length];
        parallelizeOperation((start, end) -> list2Array(list, start, end));
    }

    public Matrix flat() {
        int n = collection.length;
        int m = collection[0].length * collection[0][0].length * collection[0][0][0].length;
        double[][] result = new double[n][m];
//        parallelizeOperation((start, end) -> flat(result, start, end));

        flat(result, 0, n);
        return new Matrix(result);
    }

    private void flat(double[][] result, int start, int end) {
        int index = 0;
        for (int i = start; i < end; i++) {
            index = 0;
            for (int j = 0; j < collection[i].length; j++) {
                for (int k = 0; k < collection[i][j].length; k++) {
                    for (int l = 0; l < collection[i][j][k].length; l++) {
                        result[i][index++] = collection[i][j][k][l];
                    }
                }
            }
        }
    }

    public D3Matrix matrixPatches(Pair<Integer, Integer> window) {
        int size = getDimensionCount();
        int windowHeight = window.getValue0();

        double[][][] result = new double[size][][];
        parallelizeOperation((start, end) -> matrixPatches(result, windowHeight, start, end));

        return new D3Matrix(result);
    }

    public D3Matrix matrixPatches_test(Pair<Integer, Integer> window) {
        int size = getDimensionCount();
        int channels = getDepthCount();
        int matrixHeight = getRowCount();
        int matrixWidth = getColumnCount();
        int windowHeight = window.getValue0();
        int windowWidth = window.getValue1();

        int numOfPatches = (matrixHeight - windowHeight + 1) * (matrixWidth - windowWidth + 1);
        int windowSize = windowHeight * windowWidth;

        double[][][] result = new double[size][][];

        System.out.println(size);
        System.out.println(numOfPatches);
        System.out.println(channels);
        System.out.println(windowSize);
        System.out.println();
        matrixPatches(result, windowHeight, 0, size);


        return new D3Matrix(result);
    }

    private void matrixPatches(double[][][] result, int window, int start, int end) {
        for (int i = start; i < end; i++) {
            result[i] = getSubMatrices(collection[i], window);
        }
    }

    private double[][] getSubMatrices(double[][][] matrix, int window) {
        int channels = matrix.length;
        int windowHeight = matrix[0].length - window + 1;
        int windowWidth = matrix[0][0].length - window + 1;
        double[][] windowData = new double[channels * windowHeight * windowWidth][];
        int index = 0;
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i <= matrix[0].length - window; i++) {
                for (int j = 0; j <= matrix[0][0].length - window; j++) {
                    windowData[index++] = buildWindow(matrix, channels, i, j, window);
                }
            }
        }
        return windowData;
    }

    private double[] buildWindow(double[][][] matrix, int channels, int startRow, int startColumn, int window) {
        double[] windowData = new double[channels * window * window];
        int index = 0;
        for (int c = 0; c < channels; c++) {
            for (int i = startRow; i < startRow + window; i++) {
                for (int j = startColumn; j < startColumn + window; j++) {
                    windowData[index++] = matrix[c][i][j];
                }
            }
        }

        return windowData;
    }

    public int getDimensionCount() {
        return collection.length;
    }

    public int getDepthCount() {
        return collection[0].length;
    }

    public int getRowCount() {
        return collection[0][0].length;
    }

    public int getColumnCount() {
        return collection[0][0][0].length;
    }

    @Override
    public int getSize() {
        return getDimensionCount();
    }

    @Override
    public String shape() {
        return String.format("(%d, %d, %d, %d)", collection.length, collection[0].length,
                collection[0][0].length, collection[0][0][0].length);
    }

    @Override
    public boolean sameShape(Tensor tensor) {
        D4Matrix mat = (D4Matrix) tensor;
        return getDimensionCount() == mat.getDimensionCount() && getDepthCount() == mat.getDepthCount() &&
                getRowCount() == mat.getRowCount() && getColumnCount() == mat.getColumnCount();
    }

    public double[][][][] getMatrix() {
        return collection;
    }

    public Matrix getSubMatrix(int dim, int depth) {
        return new Matrix(collection[dim][depth]).clone();
    }

    public void setMatrix(int dim, int depth, Matrix mat) {
        collection[dim][depth] = mat.getMatrix();
    }

    private void list2Array(List<double[][][]> list, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                for (int k = 0; k < collection[i][j].length; k++) {
                    for (int l = 0; l < collection[i][j][k].length; l++) {
                        collection[i][j][k][l] = list.get(i)[j][k][l];
                    }
                }
            }
        }
    }

    public D3Matrix getDimension(int dim) {
        return new D3Matrix(collection[dim]);
    }
}
