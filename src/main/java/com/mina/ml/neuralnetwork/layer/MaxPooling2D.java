package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.*;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class MaxPooling2D extends Layer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(MaxPooling2D.class);

    private Quartet<Integer, Integer, Integer, Integer> inputShape;
    private Pair<Integer, Integer> poolSize;

    private D4WeightMatrix weight;

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

        // re-create weight matrix every time
        weight = new D4WeightMatrix(input.getDimensionCount(), input.getDepthCount(),
                input.getRowCount(), input.getColumnCount());

        int outputHeight = input.getRowCount() / poolSize.getValue0();
        int outputWidth = input.getColumnCount() / poolSize.getValue1();

        Y = new D4Matrix(input.getDimensionCount(), input.getDepthCount(), outputHeight, outputWidth);
//        Pair<Integer, Integer> maxIndices = new Pair<>(0, 0);
        for (int dim = 0; dim < input.getDimensionCount(); dim++) {
            for (int depth = 0; depth < input.getDepthCount(); depth++) {
//                Matrix maxPoolMatrix = applyPooling(input.getSubMatrix(dim, depth), maxIndices);
                Matrix maxPoolMatrix = applyPooling(dim, depth);
                Y.setMatrix(dim, depth, maxPoolMatrix);
            }
        }

//        System.out.println("MaxPooling2D Y shape = " + Y.shape());

        return Objects.isNull(nextLayer) ? Y : nextLayer.forwardPropagation(Y);
    }

    private Matrix applyPooling(int dim, int depth) {
        Matrix matrix = input.getSubMatrix(dim, depth);
        int height = matrix.getRowCount() / poolSize.getValue0();
        int width = matrix.getColumnCount() / poolSize.getValue1();

        double[][] result = new double[height][width];
        double[][] mat = matrix.getMatrix();
        int r = 0, c = 0;
        for (int i = 0; i + poolSize.getValue0() < mat.length; i += poolSize.getValue0()) {
            for (int j = 0; j + poolSize.getValue1() < mat[0].length; j += poolSize.getValue1()) {
                result[r][c++] = applyMaxPooling(mat, i, j, dim, depth);
            }
            r++;
            c = 0;
        }

        return new Matrix(result);
    }

    private double applyMaxPooling(double[][] mat, int row, int col, int dim, int depth) {
        double value = mat[row][col];
        int maxRow = row;
        int maxCol = col;

        for (int i = row; i < row + poolSize.getValue0(); i++) {
            for (int j = col; j < col + poolSize.getValue1(); j++) {
                if (mat[i][j] > value) {
                    value = mat[i][j];
                    maxRow = i;
                    maxCol = j;
                }
            }
        }

        // manually update weight
        weight.getMatrix()[dim][depth][maxRow][maxCol] = 1d;

        return value;
    }

    @Override
    public void printForwardPropagation(Tensor input) {

    }

    @Override
    public void backPropagation(Tensor costPrime) {
//        System.out.println("MaxPooling2D costPrime shape = " + costPrime.shape());
//        System.out.println("MaxPooling2D inputShape shape = " + inputShape);
//        System.out.println("MaxPooling2D poolSize shape = " + poolSize);

        D4Matrix cost = new D4Matrix(input.getDimensionCount(), input.getDepthCount());
        for (int dim = 0; dim < input.getDimensionCount(); dim++) {
            for (int depth = 0; depth < input.getDepthCount(); depth++) {
                Matrix reversePoolMatrix = applyReversePooling((D4Matrix) costPrime, dim, depth);
                cost.setMatrix(dim, depth, reversePoolMatrix);
            }
        }

//        System.out.println("MaxPooling2D cost = " + cost.shape());

        if (!Objects.isNull(prevLayer)) {
            prevLayer.backPropagation(cost);
        }

    }

    private Matrix applyReversePooling(D4Matrix cost, int dim, int depth) {
        double[][] costMatrix = cost.getSubMatrix(dim, depth).getMatrix();
        double[][] weightMatrix = weight.getSubMatrix(dim, depth).getMatrix();

        double[][] result = new double[weightMatrix.length][weightMatrix[0].length];

        Pair<Integer, Integer> indices;
        for (int i = 0; i + poolSize.getValue0()< weightMatrix.length; i += poolSize.getValue0()) {
            for (int j = 0; j + poolSize.getValue1() < weightMatrix[i].length; j += poolSize.getValue1()) {
                indices = applyReverseMaxPooling(weightMatrix, i, j);
                result[indices.getValue0()][indices.getValue1()] = costMatrix[indices.getValue0() / poolSize.getValue0()][indices.getValue1() / poolSize.getValue1()];
            }
        }

        return new Matrix(result);
    }

    private Pair<Integer, Integer> applyReverseMaxPooling(double[][] weightMatrix, int row, int col) {
        for (int i = row; i < row + poolSize.getValue0(); i++) {
            for (int j = col; j < col + poolSize.getValue1(); j++) {
                if (weightMatrix[i][j] == 1d) {
                    return new Pair<>(i, j);
                }
            }
        }

        throw new RuntimeException("Invalid Reverse Max Pooling");
    }

    @Override
    public void updateWeight(double learningRate) {
        if (!Objects.isNull(nextLayer)) {
            nextLayer.updateWeight(learningRate);
        }
    }

    @Override
    public Tensor getWeights() {
        return weight;
    }

    @Override
    public void setWeights(Tensor weight) {
        this.weight = (D4WeightMatrix) weight;
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Quartet<>(inputShape.getValue0(), inputShape.getValue1(), inputShape.getValue2() / poolSize.getValue0(), inputShape.getValue3() / poolSize.getValue1());
    }
}
