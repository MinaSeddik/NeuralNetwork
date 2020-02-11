package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.activationfunction.Sigmoid;
import com.mina.ml.neuralnetwork.activationfunction.Tansh;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.*;
import org.javatuples.Pair;
import org.javatuples.Triplet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

//Refrence: https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
//Refrence: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
//Refrence: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
//Refrence: https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/


public class LSTM extends Layer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(LSTM.class);

    private Triplet<Integer, Integer, Integer> inputShape;
    private int units;

    private WeightMatrix inputActivationWeight;
    private Matrix inputActivationDeltaWeight;
    private WeightMatrix inputGateWeight;
    private Matrix inputGateDeltaWeight;
    private WeightMatrix forgetGateWeight;
    private Matrix forgetGateDeltaWeight;
    private WeightMatrix outputGateWeight;
    private Matrix outputGateDeltaWeight;

    private WeightMatrix inputActivationUWeight;
    private Matrix inputActivationDeltaUWeight;
    private WeightMatrix inputGateUWeight;
    private Matrix inputGateDeltaUWeight;
    private WeightMatrix forgetGateUWeight;
    private Matrix forgetGateDeltaUWeight;
    private WeightMatrix outputGateUWeight;
    private Matrix outputGateDeltaUWeight;

    private BiasVector inputActivationBias;
    private Vector inputActivationDeltaBias;
    private BiasVector inputGateBias;
    private Vector inputGateDeltaBias;
    private BiasVector forgetGateBias;
    private Vector forgetGateDeltaBias;
    private BiasVector outputGateBias;
    private Vector outputGateDeltaBias;

    private D3Matrix input;
    private Matrix A;

    private String activationFunctionStr;

    private ActivationFunction tanhActivation = new Tansh();
    private ActivationFunction sigmoidActivation = new Sigmoid();

    public LSTM(int units, Tuple inputShape, String activation) {
        this(units, activation);

        switch (inputShape.getClass().getName()) {
            case "org.javatuples.Triplet":
                this.inputShape = (Triplet<Integer, Integer, Integer>) inputShape;
                break;
            default:
                RuntimeException ex = new RuntimeException("UnSupported Input Shape");
                logger.error("{}, Exception: {}", ex.getMessage(), ex);
                throw ex;
        }
    }

    public LSTM(int units, String activation) {
        this.units = units;
        activationFunctionStr = activation;
    }

    @Override
    public void buildupLayer() {
        int n = units;
        int m = inputShape.getValue2();

        // init the weight matrices and bias
        inputActivationWeight = new WeightMatrix(n, n);
        inputActivationWeight.initializeRandom(-1.0d, 1.0d);
        inputActivationDeltaWeight = new Matrix(n, n);

        inputGateWeight = new WeightMatrix(n, n);
        inputGateWeight.initializeRandom(-1.0d, 1.0d);
        inputGateDeltaWeight = new Matrix(n, n);

        forgetGateWeight = new WeightMatrix(n, n);
        forgetGateWeight.initializeRandom(-1.0d, 1.0d);
        forgetGateDeltaWeight = new Matrix(n, n);

        outputGateWeight = new WeightMatrix(n, n);
        outputGateWeight.initializeRandom(-1.0d, 1.0d);
        outputGateDeltaWeight = new Matrix(n, n);


        inputActivationUWeight = new WeightMatrix(n, m);
        inputActivationUWeight.initializeRandom(-1.0d, 1.0d);
        inputActivationDeltaUWeight = new Matrix(n, m);

        inputGateUWeight = new WeightMatrix(n, m);
        inputGateUWeight.initializeRandom(-1.0d, 1.0d);
        inputGateDeltaUWeight = new Matrix(n, m);

        forgetGateUWeight = new WeightMatrix(n, m);
        forgetGateUWeight.initializeRandom(-1.0d, 1.0d);
        forgetGateDeltaUWeight = new Matrix(n, m);

        outputGateUWeight = new WeightMatrix(n, m);
        outputGateUWeight.initializeRandom(-1.0d, 1.0d);
        outputGateDeltaUWeight = new Matrix(n, m);


        inputActivationBias = new BiasVector(n);
        inputActivationBias.initializeRandom(-1.0d, 1.0d);
        inputActivationDeltaBias = new Vector(n);

        inputGateBias = new BiasVector(n);
        inputGateBias.initializeRandom(-1.0d, 1.0d);
        inputGateDeltaBias = new Vector(n);

        forgetGateBias = new BiasVector(n);
        forgetGateBias.initializeRandom(-1.0d, 1.0d);
        forgetGateDeltaBias = new Vector(n);

        outputGateBias = new BiasVector(n);
        outputGateBias.initializeRandom(-1.0d, 1.0d);
        outputGateDeltaBias = new Vector(n);


        ActivationFunctionFactory activationFunctionFactory = new ActivationFunctionFactory();
        try {
            activationFunction = activationFunctionFactory.createActivationFunction(activationFunctionStr);
        } catch (Exception ex) {
            logger.error("{}: {}", ex.getClass(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    @Override
    public String getName() {
        return "lstm_" + layerIndex;
    }

    @Override
    public int getNumberOfParameter() {
        int n = units;
        int m = inputShape.getValue2();

        return 4 * (n * m + n * n + n);
    }

    @Override
    public Tensor forwardPropagation(Tensor inputTensor) {

        input = (D3Matrix) inputTensor;
        int numberOfSamples = input.getDepthCount();
        int timeSteps = inputShape.getValue1();

        Matrix h_prev = new Matrix(numberOfSamples, units);
        Matrix prevState = new Matrix(numberOfSamples, units);
        D3Matrix timeStepsX = reshapeInputPerTimeStep(input);

//        System.out.println("input shape = " + input.shape());
//        System.out.println("timeStepsX shape = " + timeStepsX.shape());

        for (int t = 0; t < timeSteps; t++) {
            Matrix X = timeStepsX.get(t);

            Matrix a = handleInputActivation(X, h_prev);
            Matrix i = handleInputGate(X, h_prev);
            Matrix f = handleForgetGate(X, h_prev);
            Matrix o = handleOutputGate(X, h_prev);

            System.out.println("a shape = " + a.shape());
            System.out.println("i shape = " + i.shape());
            Matrix ai = a.elementWiseProduct(i);

            System.out.println("f shape = " + f.shape());
            System.out.println("state_prev shape = " + prevState.shape());
            Matrix fs = f.elementWiseProduct(prevState);

            Matrix currentState = ai.add(fs);
            System.out.println("currentState shape = " + currentState.shape());

            A = tanhActivation.activate(currentState).elementWiseProduct(o);

            System.out.println("out shape = " + A.shape());
            prevState = currentState;
            h_prev = A;
        }

        Matrix Z = activationFunction.activate(A);
//        System.exit(0);

        return Objects.isNull(nextLayer) ? Z : nextLayer.forwardPropagation(Z);
    }

    @Override
    public void printForwardPropagation(Tensor input) {
        // not applicable
    }

    @Override
    public void backPropagation(Tensor costPrime) {

        //dE/dZ
        Matrix dE_dZ = (Matrix) costPrime;
        System.out.println("dE/dZ shape = " + dE_dZ.shape());

        // dZ/dA
        Matrix dZ_dA = activationFunction.activatePrime(A);
        System.out.println("dZ/dA shape = " + dZ_dA.shape());

        // dE/dZ
        Matrix dE_dA = dE_dZ.elementWiseProduct(dZ_dA);
        System.out.println("dE/dA shape = " + dE_dA.shape());

//        D3Matrix timeStepsX = reshapeInputPerTimeStep(input);
        int timeSteps = inputShape.getValue1();
        System.out.println("input shape shape = " + input.shape());

//        https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
        for (int sample = 0; sample < input.getDepthCount(); sample++) {
//            Vector cost = dE_dA.getRowAsVector(sample);
//            dOut = cost + zero_previous;
//            for (int t = timeSteps - 1; t >= 0; t--) {
//            }
        }


        System.exit(0);


    }

    private Matrix handleInputActivation(Matrix x, Matrix h_prev) {
        int numberOfSamples = input.getDepthCount();

//        System.out.println("X Shape = " + x.shape());
//        System.out.println("inputActivationWeight Shape = " + inputActivationUWeight.shape());
        Matrix X_Wt = x.dot(inputActivationUWeight.transpose());
//        System.out.println("a1 Shape = " + X_Wt.shape());

        Matrix H_Wt = h_prev.dot(inputActivationWeight.transpose());
//        System.out.println("a2 Shape = " + H_Wt.shape());

        Matrix bias = inputActivationBias.toMatrix().duplicateRow(numberOfSamples);
//        System.out.println("bias Shape = " + bias.shape());

        Matrix A = X_Wt.add(H_Wt).add(bias);

        return tanhActivation.activate(A);
    }

    private Matrix handleInputGate(Matrix x, Matrix h_prev) {
        int numberOfSamples = input.getDepthCount();

//        System.out.println("X Shape = " + x.shape());
//        System.out.println("inputActivationWeight Shape = " + inputGateUWeight.shape());
        Matrix X_Wt = x.dot(inputGateUWeight.transpose());
//        System.out.println("a1 Shape = " + X_Wt.shape());

        Matrix H_Wt = h_prev.dot(inputGateWeight.transpose());
//        System.out.println("a2 Shape = " + H_Wt.shape());

        Matrix bias = inputGateBias.toMatrix().duplicateRow(numberOfSamples);
//        System.out.println("bias Shape = " + bias.shape());

        Matrix A = X_Wt.add(H_Wt).add(bias);

        return sigmoidActivation.activate(A);
    }

    private Matrix handleForgetGate(Matrix x, Matrix h_prev) {
        int numberOfSamples = input.getDepthCount();

//        System.out.println("X Shape = " + x.shape());
//        System.out.println("inputActivationWeight Shape = " + forgetGateUWeight.shape());
        Matrix X_Wt = x.dot(forgetGateUWeight.transpose());
//        System.out.println("a1 Shape = " + X_Wt.shape());

        Matrix H_Wt = h_prev.dot(forgetGateWeight.transpose());
//        System.out.println("a2 Shape = " + H_Wt.shape());

        Matrix bias = forgetGateBias.toMatrix().duplicateRow(numberOfSamples);
//        System.out.println("bias Shape = " + bias.shape());

        Matrix A = X_Wt.add(H_Wt).add(bias);

        return sigmoidActivation.activate(A);
    }

    private Matrix handleOutputGate(Matrix x, Matrix h_prev) {
        int numberOfSamples = input.getDepthCount();

//        System.out.println("X Shape = " + x.shape());
//        System.out.println("inputActivationWeight Shape = " + outputGateUWeight.shape());
        Matrix X_Wt = x.dot(outputGateUWeight.transpose());
//        System.out.println("a1 Shape = " + X_Wt.shape());

        Matrix H_Wt = h_prev.dot(outputGateWeight.transpose());
//        System.out.println("a2 Shape = " + H_Wt.shape());

        Matrix bias = outputGateBias.toMatrix().duplicateRow(numberOfSamples);
//        System.out.println("bias Shape = " + bias.shape());

        Matrix A = X_Wt.add(H_Wt).add(bias);

        return sigmoidActivation.activate(A);
    }

    private D3Matrix reshapeInputPerTimeStep(D3Matrix input) {
        return input.swapDepthAndColumns();
    }

    @Override
    public void updateWeight(double learningRate) {
//        W.updateWeights(dW, learningRate);
//        U.updateWeights(dU, learningRate);
//        B.updateBias(dB, learningRate);

        if (!Objects.isNull(nextLayer)) {
            nextLayer.updateWeight(learningRate);
        }
    }

    // todo bug we have W, U and bias
    @Override
    public Tensor getWeights() {
        return null;
    }

    // todo bug we have W, U and bias
    @Override
    public void setWeights(Tensor weight) {
        this.inputActivationDeltaUWeight = (WeightMatrix) weight;
    }

    @Override
    public void setInputShape(Tuple inputShape) {
        this.inputShape = (Triplet<Integer, Integer, Integer>) inputShape;
    }

    @Override
    public Tuple getOutputShape() {
        return new Pair<>(inputShape.getValue0(), units);
    }
}
