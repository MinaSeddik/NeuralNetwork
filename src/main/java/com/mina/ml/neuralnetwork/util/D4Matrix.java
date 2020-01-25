package com.mina.ml.neuralnetwork.util;

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
        return getDepthCount();
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
