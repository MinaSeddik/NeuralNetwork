package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class D4FeatureMatrix extends D4Matrix {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(D4FeatureMatrix.class);

    public D4FeatureMatrix(int size) {
        super(size, 0, 0, 0);
    }

    public D4FeatureMatrix(int dimension, int depth, int rows, int columns) {
        super(dimension, depth, rows, columns);
    }

    public D4FeatureMatrix(double[][][][] d4Matrix) {
        super(d4Matrix);
    }

    public D4Matrix buildFeatures(D4Matrix X, D4Matrix weight, Pair<Integer, Integer> kernelSize,
                                  int filters, int outputHeight, int outputWidth, Vector bias) {
        double[][][][] result = new double[X.getDimensionCount()][filters][outputHeight][outputWidth];
        parallelizeOperation((start, end) -> buildFeatures(result, X, weight, kernelSize,
                filters, outputHeight, outputWidth, bias, start, end));

        return new D4Matrix(result);
    }


    private void buildFeatures(double[][][][] result, D4Matrix X, D4Matrix weight, Pair<Integer, Integer> kernelSize,
                               int filters, int outputHeight, int outputWidth, Vector bias,
                               int startIndex, int endIndex) {

        for (int n = startIndex; n < endIndex; n++) {
            for (int f = 0; f < filters; f++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        result[n][f][i][j] = getFeature(X.getDimension(n), weight, kernelSize, f, i, j)
                                + bias.getElement(f);
                    }
                }
            }
        }
    }

    private double getFeature(D3Matrix X, D4Matrix weight, Pair<Integer, Integer> kernelSize,
                              int filterId, int i, int j) {
        assert X.getDepthCount() == weight.getDimensionCount();

        double feature = 0d;
        int channels = X.getDepthCount();
        int kernalHeight = kernelSize.getValue0();
        int kernalWidth = kernelSize.getValue1();

        double[][][] x = X.getMatrix();
        double[][][][] w = weight.getMatrix();

        for (int c = 0; c < channels; c++) {
            for (int k = 0; k < kernalHeight; k++) {
                for (int l = 0; l < kernalWidth; l++) {
                    feature += w[filterId][c][k][l] * x[c][i + k][j + l];
                }
            }
        }

        return feature;

    }

}
