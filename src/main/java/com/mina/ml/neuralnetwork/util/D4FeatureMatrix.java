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

    public D4Matrix calculateOutputPrime(D4Matrix dError, D4Matrix X, D4Matrix weight, Pair<Integer, Integer> kernelSize,
                                         int filters, int padding) {
        int numberOfSamples = X.getDimensionCount();
        int h = X.getRowCount() + 2 * padding;
        int w = X.getColumnCount() + 2 * padding;
        int channels = X.getDepthCount();
        int kernalHeight = kernelSize.getValue0();
        int kernalWidth = kernelSize.getValue1();

        int zeroPaddingH = kernalHeight - 1;
        int zeroPaddingW = kernalWidth - 1;

        double[][][][] output = dError.addZeroPadding(zeroPaddingH, zeroPaddingW).getMatrix();
        double[][][][] weightT = weight.transpose().getMatrix();

        double[][][][] result = new double[numberOfSamples][channels][h][w];
        parallelizeOperation((start, end) -> calculateOutputPrime(result, output, X, weightT, kernelSize,
                filters, padding, start, end));

        return new D4Matrix(result);
    }

    private void calculateOutputPrime(double[][][][] result, double[][][][] output, D4Matrix X, double[][][][] weightT,
                                      Pair<Integer, Integer> kernelSize,
                                      int filters, int padding,
                                      int startIndex, int endIndex) {

        int h = X.getRowCount() + 2 * padding;
        int w = X.getColumnCount() + 2 * padding;
        int kernalHeight = kernelSize.getValue0();
        int kernalWidth = kernelSize.getValue1();
        int channels = X.getDepthCount();

        for (int n = startIndex; n < endIndex; n++) {
            for (int f = 0; f < filters; f++) {
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        for (int k = 0; k < kernalHeight; k++) {
                            for (int l = 0; l < kernalWidth; l++) {
                                for (int c = 0; c < channels; c++) {
                                    double dx = i + k < h && j + l < w ? output[n][f][i + k][j + l] : 0d;
                                    result[n][c][i][j] += dx * weightT[f][c][k][l];
                                }
                            }
                        }
                    }
                }
            }
        }

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
