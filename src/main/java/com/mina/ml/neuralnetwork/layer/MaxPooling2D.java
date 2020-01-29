package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.D4Matrix;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class MaxPooling2D extends Layerrr {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(MaxPooling2D.class);

    private Quartet<Integer, Integer, Integer, Integer> inputShape;
    private Pair<Integer, Integer> poolSize;

    private D4Matrix input;
    private D4Matrix Y;

    public MaxPooling2D(Pair<Integer, Integer> poolSize) {
        this.poolSize = poolSize;
    }

    @Override
    public void buildupLayer() {
        assert poolSize.getValue0() == poolSize.getValue1();
    }

    @Override
    public String getName() {
        return "max_pooling2d_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        return 0;
    }

    @Override
    public Tensor forwardPropagation(Tensor inputTensor) {

//        System.out.println("MaxPooling2D inputShape " + inputShape);

        input = (D4Matrix) inputTensor;
        int outputHeight = input.getRowCount() / poolSize.getValue0();
        int outputWidth = input.getColumnCount() / poolSize.getValue0();

        Y = new D4Matrix(input.getDimensionCount(), input.getDepthCount(), outputHeight, outputWidth);
        for (int dim = 0; dim < input.getDimensionCount(); dim++) {
            for (int depth = 0; depth < input.getDepthCount(); depth++) {
                Matrix maxPoolMatrix = applyPooling(input.getSubMatrix(dim, depth));
                Y.setMatrix(dim, depth, maxPoolMatrix);
            }
        }

//        System.out.println("MaxPooling2D Y shape = " + Y.shape());

        return Objects.isNull(nextLayer) ? Y : nextLayer.forwardPropagation(Y);
    }

    private Matrix applyPooling(Matrix matrix) {
        int height = matrix.getRowCount() / 2;
        int width = matrix.getColumnCount() / 2;

        double[][] result = new double[height][width];
        double[][] mat = matrix.getMatrix();
        int r = 0, c = 0;
        for (int i = 0; i < mat.length; i += poolSize.getValue0()) {
            for (int j = 0; j < mat[0].length; j += poolSize.getValue0()) {
                result[r][c++] = applyMaxPooling(mat, i, j);
            }
            r++;
            c = 0;
        }

        return new Matrix(result);
    }

    private double applyMaxPooling(double[][] mat, int row, int col) {
        double value = mat[row][col];

        for (int i = row; i < row + poolSize.getValue0(); i++) {
            for (int j = col; j < col + poolSize.getValue1(); j++) {
                if (mat[i][j] > value) {
                    value = mat[i][j];
                }
            }
        }

        return value;
    }

    @Override
    public void printForwardPropagation(Tensor input) {

    }

    @Override
    public void backPropagation(Tensor costPrime) {

    }

    @Override
    public void updateWeight(double learningRate) {

    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public void setWeights(Tensor weight) {

    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Quartet<>(inputShape.getValue0(), inputShape.getValue1(), inputShape.getValue2() / 2, inputShape.getValue3() / 2);
    }
}
