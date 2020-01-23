package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class D3WeightMatrix extends D3Matrix {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(D3WeightMatrix.class);

    public D3WeightMatrix(int depth, int rows, int columns) {
        super(depth, rows, columns);
    }

    public D3WeightMatrix(double[][][] d3Matrix) {
        super(d3Matrix);
    }


}
