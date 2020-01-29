package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class D3Matrix extends Tensor {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

    protected double[][][] collection;

    public D3Matrix(int depth, int rows, int columns) {
        collection = new double[depth][rows][columns];
    }

    public D3Matrix(double[][][] d3Matrix) {
        collection = d3Matrix;
    }

    public D3Matrix(List<double[][]> list) {
        collection = new double[list.size()][list.get(0).length][list.get(0)[0].length];
        parallelizeOperation((start, end) -> list2Array(list, start, end));
    }

    public int getDepthCount() {
        return collection.length;
    }

    public int getRowCount() {
        return collection[0].length;
    }

    public int getColumnCount() {
        return collection[0][0].length;
    }

    public Matrix get(int index){
        return new Matrix(collection[index]);
    }

    public double[][][] getMatrix() {
        return collection;
    }

    public Matrix flat(){
        int n = collection.length;
        int m = collection[0].length * collection[0][0].length;
        double[][] result = new double[n][m];
        parallelizeOperation((start, end) -> flat(result, start, end));

        return new Matrix(result);
    }

    private void flat(double[][] result, int start, int end) {
        int index;
        for (int i = start; i < end; i++) {
            index = 0;
            for(int j=0;j<collection[i].length;j++){
                for(int k=0;k<collection[i][j].length;k++){
                    result[i][index++] = collection[i][j][k];
                }
            }
        }
    }

    @Override
    public int getSize() {
        return getDepthCount();
    }

    @Override
    public String shape() {
        return String.format("(%d, %d, %d)", collection.length, collection[0].length,
                collection[0][0].length);
    }

    @Override
    public boolean sameShape(Tensor tensor) {
        D3Matrix mat = (D3Matrix) tensor;
        return getDepthCount() == mat.getDepthCount() &&
                getRowCount() == mat.getRowCount() && getColumnCount() == mat.getColumnCount();
    }

    private void list2Array(List<double[][]> list, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                for (int k = 0; k < collection[i][j].length; k++) {
                    collection[i][j][k] = list.get(i)[j][k];
                }
            }
        }
    }
}
