package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.WeightMatrix;
import org.javatuples.Tuple;
import org.javatuples.Unit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

public class Dense extends Layerrr {

    private final static Logger logger = LoggerFactory.getLogger(Dense.class);

    protected WeightMatrix weight;
    protected Matrix deltaWeight;

    private String activationFunctionStr;

    public Dense(int units, Tuple inputShape, String activation) {
        this(units, activation);

        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Unit":
                numOfInputs = ((Unit<Integer>) inputShape).getValue0();
                numOfInputs += 1; // Add 1 for Bias
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
    }

    public Dense(int units, String activation) {
        numOfOutputs = units;
        activationFunctionStr = activation;
    }

    public void buildupLayer() {

        // init the weight matrix
        weight = new WeightMatrix(numOfInputs, numOfOutputs);
        weight.initializeRandom(-1.0d, 1.0d);

        deltaWeight = new Matrix(numOfInputs, numOfOutputs);

        ActivationFunctionFactory activationFunctionFactory = new ActivationFunctionFactory();
        try {
            activationFunction = activationFunctionFactory.createActivationFunction(activationFunctionStr);
        } catch (Exception ex) {
            logger.error("{}: {}", ex.getClass(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    private Matrix addBias(Matrix matrix) {
        return matrix.addColumn(1d);
    }

    @Override
    public Matrix forwardPropagation(Matrix input) {
//        System.out.println("------------------------");
//        System.out.println("Input Shape: " + input.shape());
        String printedMatrix = MatrixManipulator.simulate("Input " + input.shape(), input.getRowCount(), input.getColumnCount(), 'b', 'x');
        System.out.println(printedMatrix);
        System.exit(0);
        input = addBias(input);
//        System.out.println("Input Shape (After Bias): " + input.shape());
//        System.out.println("weight Shape: " + weight.shape());
        Matrix A = input.dot(weight);
//        System.out.println("weight A: " + A.shape());
        Matrix Z = activationFunction.activate(A);

        return Objects.isNull(nextDense) ? Z : nextDense.forwardPropagation(Z);
    }

    @Override
    public void backPropagation(Matrix costPrime){
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

    }

    @Override
    public String getName() {
        return "dense_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        return numOfInputs * numOfOutputs;
    }

}
