package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class D4WeightMatrix extends D4Matrix {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(D4WeightMatrix.class);

    public D4WeightMatrix(int dimension, int depth, int rows, int columns) {
        super(dimension, depth, rows, columns);
    }

    public D4WeightMatrix(double[][][][] d4Matrix) {
        super(d4Matrix);
    }


}
