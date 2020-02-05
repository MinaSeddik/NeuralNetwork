package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.D3Matrix;
import com.mina.ml.neuralnetwork.util.D4Matrix;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.Tensor;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Triplet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class Flatten extends Layer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Flatten.class);

    private Tuple inputShape;

    @Override
    public void buildupLayer() {
        // NOTHING
    }

    @Override
    public String getName() {
        return "flatten_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        return 0;
    }

    @Override
    public Tensor forwardPropagation(Tensor inputTensor) {

//        System.out.println("Flatten inputShape " + inputShape);

        Matrix Y;
        if (inputTensor instanceof Matrix) {
            Y = (Matrix) inputTensor;
        } else if (inputTensor instanceof D3Matrix) {
            D3Matrix input = (D3Matrix) inputTensor;
            Y = input.flat();
        } else if (inputTensor instanceof D4Matrix) {
            D4Matrix input = (D4Matrix) inputTensor;
            Y = input.flat();
        } else {
            RuntimeException ex = new RuntimeException("UnSupported Input Shape");
            logger.error("{}, Exception: {}", ex.getMessage(), ex);
            throw ex;
        }

//        System.out.println("Flatten Y shape = " + Y.shape());
//        System.exit(0);

        return Objects.isNull(nextLayer) ? Y : nextLayer.forwardPropagation(Y);
    }

    @Override
    public void printForwardPropagation(Tensor input) {

    }

    @Override
    public void backPropagation(Tensor costPrime) {
//        System.out.println("Flatten costPrime shape = " + costPrime.shape());
//        System.out.println("Flatten inputShape shape = " + inputShape);

        Matrix costPrimeMatrix = (Matrix) costPrime;
        Tensor cost;
        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Pair":
                cost = costPrimeMatrix;
                break;
            case "org.javatuples.Triplet":
                Triplet<Integer, Integer, Integer> shape3 = (Triplet<Integer, Integer, Integer>) inputShape;
                D3Matrix d3Matrix = new D3Matrix(costPrimeMatrix.getRowCount(), shape3.getValue1(), shape3.getValue2());
                cost = d3Matrix.reshape(costPrimeMatrix);
                break;
            case "org.javatuples.Quartet":
                Quartet<Integer, Integer, Integer, Integer> shape4 = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
                D4Matrix d4Matrix = new D4Matrix(costPrimeMatrix.getRowCount(), shape4.getValue1(), shape4.getValue2(), shape4.getValue3());
                cost = d4Matrix.reshape(costPrimeMatrix);
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }

//        System.out.println("cost.shape(): " + cost.shape());

        if (!Objects.isNull(prevLayer)) {
            prevLayer.backPropagation(cost);
        }
    }

    @Override
    public void updateWeight(double learningRate) {
        if (!Objects.isNull(nextLayer)) {
            nextLayer.updateWeight(learningRate);
        }
    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public void setWeights(Tensor weight) {
        // N/A
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Pair<>(0, getNumberOfOutput());
    }

    private int getNumberOfOutput() {
        int links;
        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Pair":
                links = ((Pair<Integer, Integer>) inputShape).getValue0();
                break;
            case "org.javatuples.Triplet":
                Triplet<Integer, Integer, Integer> shape3 = (Triplet<Integer, Integer, Integer>) inputShape;
                links = shape3.getValue1() * shape3.getValue2();
                break;
            case "org.javatuples.Quartet":
                Quartet<Integer, Integer, Integer, Integer> shape4 = (Quartet<Integer, Integer, Integer, Integer>) inputShape;
                links = shape4.getValue1() * shape4.getValue2() * shape4.getValue3();
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }

        return links;
    }
}
