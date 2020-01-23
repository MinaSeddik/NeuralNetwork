package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class D3Matrix extends Tensor{

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

    protected double[][][] collection;

    public D3Matrix(int depth, int rows, int columns) {
        collection = new double[depth][rows][columns];
    }

    public D3Matrix(double[][][] d3Matrix) {
        collection = d3Matrix;
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

    @Override
    public int getSize() {
        return getDepthCount();
    }


}
