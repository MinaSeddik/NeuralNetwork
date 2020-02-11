package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.MatrixLogger;
import com.mina.ml.neuralnetwork.util.Tensor;
import com.mina.ml.neuralnetwork.util.WeightMatrix;
import org.javatuples.Pair;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Dense extends Layer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Dense.class);

    private Pair<Integer, Integer> inputShape;
    private int units; // number of neurons

    private WeightMatrix weight;
    private Matrix deltaWeight;

    private Matrix input;
    private Matrix A;

    private String activationFunctionStr;

    public Dense(int units, Tuple inputShape, String activation) {
        this(units, activation);

        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Pair":
                this.inputShape = (Pair<Integer, Integer>)inputShape;
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
    }

    public Dense(int units, String activation) {
        this.units = units;
        activationFunctionStr = activation;
    }

    @Override
    public void buildupLayer() {

        // init the weight matrix
        weight = new WeightMatrix(inputShape.getValue1() + 1 /* 1 for Bias */, units);
        weight.initializeRandom(-1.0d, 1.0d);

        deltaWeight = new Matrix(inputShape.getValue1() + 1 /* 1 for Bias */, units);

        ActivationFunctionFactory activationFunctionFactory = new ActivationFunctionFactory();
        try {
            activationFunction = activationFunctionFactory.createActivationFunction(activationFunctionStr);
        } catch (Exception ex) {
            logger.error("{}: {}", ex.getClass(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    private Matrix addBias(Tensor matrix) {
        return ((Matrix)matrix).addColumn(1d);
    }

    private Matrix removeBias(Matrix matrix) {
        return matrix.removeFirstColumn();
    }

    @Override
    public void printForwardPropagation(Tensor input) {

        Matrix tempInput = ((Matrix)input).clone();
        Matrix tempWeight = weight.clone();

        System.out.println(String.format("Layer %s", getName()));
        String printedMatrix = MatrixLogger.simulate("Input " + tempInput.shape(), tempInput.getRowCount(), tempInput.getColumnCount(), 1, 'b', 'x');
        System.out.println(printedMatrix);

        tempWeight = addBias(tempWeight);

        printedMatrix = MatrixLogger.simulate("Input - after adding Bias [x0=1] - " + tempInput.shape(), tempInput.getRowCount(), tempWeight.getColumnCount(), 0, 'b', 'x');
        System.out.println(printedMatrix);

        printedMatrix = MatrixLogger.simulate("Weights " + tempWeight.shape(), tempWeight.getRowCount(), tempWeight.getColumnCount(), 0, 'w', Character.MIN_VALUE);
        System.out.println(printedMatrix);

        System.out.println(String.format("A = Input %s * weight %s", tempInput.shape(), tempWeight.shape()));
        // take care, the input will change
        Matrix A = tempInput.dot(tempWeight);
        System.out.println(String.format("A %s", A.shape()));

        Matrix Z = activationFunction.activate(A);
        System.out.println(String.format("Z %s = activate(A)", Z.shape()));

        System.out.println("---------------------------------------");

        if (!Objects.isNull(nextLayer)) {
            nextLayer.printForwardPropagation(Z);
        }

    }

    @Override
    public Tensor forwardPropagation(Tensor in) {
        input = addBias(in);
        A = input.dot(weight);
        Matrix Z = activationFunction.activate(A);

        return Objects.isNull(nextLayer) ? Z : nextLayer.forwardPropagation(Z);
    }

    @Override
    public void backPropagation(Tensor costPrime) {
//        /* output */
//        calculateDeltaWeight(costOutputPrime);
//        prepareErrorCostThenBackPropagate(costOutputPrime);
//
//        /* hidden */
//        double[][] primeA = activationFunction.activatePrime(A);
//        double[][] costOutputPrime = MatrixManipulator.multiplyEntries(prevCostPrime, primeA);
//        calculateDeltaWeight(costOutputPrime);
//        prepareErrorCostThenBackPropagate(costOutputPrime);

        /* input */
        // NOTHING

        //----------------------------------
        //dE/dZ
        Matrix dE_dZ = (Matrix) costPrime;
//        System.out.println("dE/dZ shape = " + dE_dZ.shape());

        // dZ/dA
        Matrix dZ_dA = activationFunction.activatePrime(A);
//        System.out.println("dZ/dA shape = " + dZ_dA.shape());

        // dE/dZ
        Matrix dE_dA = dE_dZ.elementWiseProduct(dZ_dA);
//        System.out.println("dE/dA shape = " + dE_dA.shape());
//
//        System.out.println("Input shape = " + input.shape());
//        System.out.println("weight shape = " + weight.shape());
//        System.out.println("deltaWeight shape = " + deltaWeight.shape());

//        List<Matrix> deltas = Collections.synchronizedList(new ArrayList<Matrix>());

        // Calculate delta-Weight (dE/dW)
//        deltaWeight.reset();
        List<Matrix> deltas = IntStream.range(0, input.getRowCount())
                .parallel()
                .mapToObj(i -> input.getRowAsVector(i)
                        .toMatrix()
                        .transpose()
                        .dot(dE_dA.getRowAsVector(i)
                                .toMatrix()))
                .collect(Collectors.toList());
        Matrix totalDeltas = accumulateDeltaWeigh(deltas);
//                .forEach(dw -> deltas.add(dw));
//                .forEach(dw -> accumulateDeltaWeigh(dw));
//        System.out.println(deltas.size());
        deltaWeight = totalDeltas.divide(input.getRowCount());

        if (!Objects.isNull(prevLayer)) {
            Matrix weightT = weight.transpose();
            Matrix cost = dE_dA.dot(weightT);
            cost = removeBias(cost);

            prevLayer.backPropagation(cost);
        }

    }

    @Override
    public void updateWeight(double learningRate) {
        weight.updateWeights(deltaWeight, learningRate);
        if (!Objects.isNull(nextLayer)) {
            nextLayer.updateWeight(learningRate);
        }
    }

    private Matrix accumulateDeltaWeigh(List<Matrix> deltas) {
        return new Matrix(deltas.get(0).getRowCount(), deltas.get(0).getColumnCount())
                .addMatrices(deltas);
    }

    public Tensor getWeights() {
        return weight;
    }

    public void setWeights(Tensor weight) {
        this.weight = (WeightMatrix) weight;
    }

    @Override
    public String getName() {
        return "dense_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        return (inputShape.getValue1() + 1 )* units;
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Pair<Integer, Integer>)inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Pair<>(inputShape.getValue0(), units);
    }

}
