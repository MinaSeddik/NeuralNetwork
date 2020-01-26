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

    public D4Matrix matrixPatches(D4Matrix matrix, Pair<Integer, Integer> window) {
        int size = matrix.getDimensionCount();
        int channels = matrix.getDepthCount();
        int matrixHeight = matrix.getRowCount();
        int matrixWidth = matrix.getColumnCount();
        int windowHeight = window.getValue0();
        int windowWidth = window.getValue1();

        int numOfPatches = (matrixHeight - windowHeight + 1) * (matrixWidth - windowWidth + 1);
        int windowSize = windowHeight * windowWidth;

        double[][][][] result = new double[size][numOfPatches][channels][windowSize];
        parallelizeOperation((start, end) -> matrixPatches(result, windowHeight, start, end));


        return new D4Matrix(result);
    }

    private void matrixPatches(double[][][][] result, int window, int start, int end) {
        int patch = 0;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                double[][] patches = getSubMatrices(collection[i][j], window);
                for (int x = 0; x < patches.length; ++x) {
                    result[i][patch++][j] = patches[x];
                }
            }
        }
    }

    private double[][] getSubMatrices(double[][] matrix, int window) {

//        https://algorithms.tutorialhorizon.com/sliding-window-algorithm-track-the-maximum-of-each-subarray-of-size-k/
//        for(int i=0;i<matrix.length;i++){
//            for(int j=0;j<matrix[0].length;j++){
//
//            }
//        }

        return null;
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

}
