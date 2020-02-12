package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.activationfunction.Sigmoid;
import com.mina.ml.neuralnetwork.activationfunction.Tansh;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.Vector;
import com.mina.ml.neuralnetwork.util.*;
import org.javatuples.Pair;
import org.javatuples.Triplet;
import org.javatuples.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

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

    private Map<Integer, Matrix> inputActivationMap = new HashMap<>();
    private Map<Integer, Matrix> inputGateMap = new HashMap<>();
    private Map<Integer, Matrix> forgetGateMap = new HashMap<>();
    private Map<Integer, Matrix> outputGateMap = new HashMap<>();
    private Map<Integer, Matrix> stateMap = new HashMap<>();
    private Map<Integer, Matrix> outputMap = new HashMap<>();

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

        // some clean-up
        inputActivationMap.clear();
        inputGateMap.clear();
        forgetGateMap.clear();
        outputGateMap.clear();
        stateMap.clear();

//        System.out.println("input shape = " + input.shape());
//        System.out.println("timeStepsX shape = " + timeStepsX.shape());

        for (int t = 0; t < timeSteps; t++) {
            Matrix X = timeStepsX.get(t);

            Matrix a = handleInputActivation(X, h_prev);
            Matrix i = handleInputGate(X, h_prev);
            Matrix f = handleForgetGate(X, h_prev);
            Matrix o = handleOutputGate(X, h_prev);

//            System.out.println("a shape = " + a.shape());
//            System.out.println("i shape = " + i.shape());
            Matrix ai = a.elementWiseProduct(i);

//            System.out.println("f shape = " + f.shape());
//            System.out.println("state_prev shape = " + prevState.shape());
            Matrix fs = f.elementWiseProduct(prevState);

            Matrix state = ai.add(fs);
//            System.out.println("state shape = " + state.shape());

            A = tanhActivation.activate(state).elementWiseProduct(o);

            // save the gates and state for back-propagation
            inputActivationMap.put(t, a);
            inputGateMap.put(t, i);
            forgetGateMap.put(t, f);
            outputGateMap.put(t, o);
            stateMap.put(t, state);
            outputMap.put(t, A);

//            System.out.println("out shape = " + A.shape());
            prevState = state;
            h_prev = A;
        }

//        System.out.println("will exit in forward ....");
//        System.exit(0);

        Matrix Z = activationFunction.activate(A);

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
//        System.out.println("dE/dZ shape = " + dE_dZ.shape());

        // dZ/dA
        Matrix dZ_dA = activationFunction.activatePrime(A);
//        System.out.println("dZ/dA shape = " + dZ_dA.shape());

        // dE/dZ
        Matrix dE_dA = dE_dZ.elementWiseProduct(dZ_dA);
//        System.out.println("dE/dA shape = " + dE_dA.shape());

//        D3Matrix timeStepsX = reshapeInputPerTimeStep(input);
        int numberOfSamples = input.getDepthCount();
        int timeSteps = inputShape.getValue1();
//        System.out.println("input shape shape = " + input.shape());
        D3Matrix timeStepsX = reshapeInputPerTimeStep(input);

        Matrix ones = new Matrix(numberOfSamples, units);
        ones.initialize(1d);
        Matrix zeros = new Matrix(numberOfSamples, units);
        ones.initialize(0d);

        Matrix cost = dE_dA;
        Matrix futureOut = new Matrix(numberOfSamples, units);
        Matrix futureState = new Matrix(numberOfSamples, units);
        for (int t = timeSteps - 1; t >= 0; t--) {
            Matrix dout = cost.add(futureOut);
            Matrix temp = ones.subtract(tanhActivation.activate(stateMap.get(t)).square());
            Matrix futureForget = forgetGateMap.containsKey(t + 1) ? forgetGateMap.get(t + 1) : zeros;

            Matrix dState = dout.elementWiseProduct(outputGateMap.get(t))
                    .elementWiseProduct(temp)
                    .add(futureState.elementWiseProduct(futureForget));

            Matrix dA = handleInputActivationPrime(dState, inputGateMap.get(t), inputActivationMap.get(t));
            Matrix dI = handleInputGatePrime(dState, inputActivationMap.get(t), inputGateMap.get(t));

            Matrix prevState = stateMap.containsKey(t - 1) ? stateMap.get(t - 1) : zeros;
            Matrix dF = handleForgetGatePrime(dState, prevState, forgetGateMap.get(t));
            Matrix dO = handleOutputGatePrime(dout, stateMap.get(t), outputGateMap.get(t));

//            System.out.println("inputActivationWeight shape = " + inputActivationWeight.shape());
//            System.out.println("inputActivationUWeight shape = " + inputActivationUWeight.shape());

            List<Vector> out = new ArrayList<>();
            Matrix features = timeStepsX.get(t);

            Matrix out_t = t == 0 ? zeros : outputMap.get(t - 1);

            Matrix inputActivation_DeltaUWeight = new Matrix(units, inputShape.getValue2());
            Matrix inputGate_DeltaUWeight = new Matrix(units, inputShape.getValue2());
            Matrix forgetGate_DeltaUWeight = new Matrix(units, inputShape.getValue2());
            Matrix outputGate_DeltaUWeight = new Matrix(units, inputShape.getValue2());

            Matrix inputActivation_DeltaWeight = new Matrix(units, units);
            Matrix inputGate_DeltaWeight = new Matrix(units, units);
            Matrix forgetGate_DeltaWeight = new Matrix(units, units);
            Matrix outputGate_DeltaWeight = new Matrix(units, units);

            Vector inputActivation_DeltaBias = new Vector(units);
            Vector inputGate_DeltaBias = new Vector(units);
            Vector forgetGate_DeltaBias = new Vector(units);
            Vector outputGate_DeltaBias = new Vector(units);

            for (int i = 0; i < numberOfSamples; i++) {
//                System.out.println("features shape = " + features.shape());
                Matrix x = features.getRowAsVector(i).toMatrix();

                Matrix dA_sample = dA.getRowAsVector(i).toMatrix();
                Matrix dI_sample = dI.getRowAsVector(i).toMatrix();
                Matrix dF_sample = dF.getRowAsVector(i).toMatrix();
                Matrix dO_sample = dO.getRowAsVector(i).toMatrix();

                Matrix dGateStateA = dA_sample.dot(inputActivationWeight.transpose());
                Matrix dGateStateI = dI_sample.dot(inputGateWeight.transpose());
                Matrix dGateStateF = dF_sample.dot(forgetGateWeight.transpose());
                Matrix dGateStateO = dO_sample.dot(outputGateWeight.transpose());

                // calculate out cost to propagate
                out.add(dGateStateA.add(dGateStateI).add(dGateStateF).add(dGateStateO).getRowAsVector(0));

//                System.out.println("**** dA_sample shape = " + dA_sample.shape());

                // errors in add
                inputActivation_DeltaUWeight = inputActivation_DeltaUWeight.add(dA_sample.transpose().dot(x));
                inputGate_DeltaUWeight = inputGate_DeltaUWeight.add(dI_sample.transpose().dot(x));
                forgetGate_DeltaUWeight = forgetGate_DeltaUWeight.add(dF_sample.transpose().dot(x));
                outputGate_DeltaUWeight = outputGate_DeltaUWeight.add(dO_sample.transpose().dot(x));

//                System.out.println("inputActivation_DeltaUWeight shape = " + inputActivation_DeltaUWeight.shape());
//                System.out.println("inputGate_DeltaUWeight shape = " + inputGate_DeltaUWeight.shape());
//                System.out.println("forgetGate_DeltaUWeight shape = " + forgetGate_DeltaUWeight.shape());
//                System.out.println("outputGate_DeltaUWeight shape = " + outputGate_DeltaUWeight.shape());


                Matrix currentSample = out_t.getRowAsVector(i).toMatrix();
                inputActivation_DeltaWeight = inputActivation_DeltaWeight.add(dA_sample.transpose().dot(currentSample));
                inputGate_DeltaWeight = inputGate_DeltaWeight.add(dI_sample.transpose().dot(currentSample));
                forgetGate_DeltaWeight = forgetGate_DeltaWeight.add(dF_sample.transpose().dot(currentSample));
                outputGate_DeltaWeight = outputGate_DeltaWeight.add(dO_sample.transpose().dot(currentSample));

//                System.out.println("inputActivation_DeltaWeight shape = " + inputActivation_DeltaWeight.shape());
//                System.out.println("inputGate_DeltaWeight shape = " + inputGate_DeltaWeight.shape());
//                System.out.println("forgetGate_DeltaWeight shape = " + forgetGate_DeltaWeight.shape());
//                System.out.println("outputGate_DeltaWeight shape = " + outputGate_DeltaWeight.shape());


                inputActivation_DeltaBias = inputActivation_DeltaBias.add(dA.getRowAsVector(i));
                inputGate_DeltaBias = inputGate_DeltaBias.add(dI.getRowAsVector(i));
                forgetGate_DeltaBias = forgetGate_DeltaBias.add(dF.getRowAsVector(i));
                outputGate_DeltaBias = outputGate_DeltaBias.add(dO.getRowAsVector(i));

//                System.out.println("inputActivation_DeltaBias shape = " + inputActivation_DeltaBias.shape());
//                System.out.println("inputGate_DeltaBias shape = " + inputGate_DeltaBias.shape());
//                System.out.println("forgetGate_DeltaBias shape = " + forgetGate_DeltaBias.shape());
//                System.out.println("outputGate_DeltaBias shape = " + outputGate_DeltaBias.shape());

            } // end loop on samples

            futureState = dState;
            futureOut = new Matrix(out.stream().map(v -> v.asArray()).collect(Collectors.toList()));

            // update delta average weights and bias
            inputActivationDeltaUWeight = inputActivationDeltaUWeight.add(inputActivation_DeltaUWeight.divide(numberOfSamples));
            inputGateDeltaUWeight = inputGateDeltaUWeight.add(inputGate_DeltaUWeight.divide(numberOfSamples));
            forgetGateDeltaUWeight = forgetGateDeltaUWeight.add(forgetGate_DeltaUWeight.divide(numberOfSamples));
            outputGateDeltaUWeight = outputGateDeltaUWeight.add(outputGate_DeltaUWeight.divide(numberOfSamples));

            inputActivationDeltaWeight = inputActivationDeltaWeight.add(inputActivation_DeltaWeight.divide(numberOfSamples));
            inputGateDeltaWeight = inputGateDeltaWeight.add(inputGate_DeltaWeight.divide(numberOfSamples));
            forgetGateDeltaWeight = forgetGateDeltaWeight.add(forgetGate_DeltaWeight.divide(numberOfSamples));
            outputGateDeltaWeight = outputGateDeltaWeight.add(outputGate_DeltaWeight.divide(numberOfSamples));

            inputActivationDeltaBias = inputActivationDeltaBias.add(inputActivation_DeltaBias.divide(numberOfSamples));
            inputGateDeltaBias = inputGateDeltaBias.add(inputGate_DeltaBias.divide(numberOfSamples));
            forgetGateDeltaBias = forgetGateDeltaBias.add(forgetGate_DeltaBias.divide(numberOfSamples));
            outputGateDeltaBias = outputGateDeltaBias.add(outputGate_DeltaBias.divide(numberOfSamples));

        }


//        System.out.println("will back propagate futureOut shape = " + futureOut.shape());
//        System.out.println("Done ...");
//        System.exit(0);

        if (!Objects.isNull(prevLayer)) {
            prevLayer.backPropagation(futureOut);
        }

    }

    private Matrix handleOutputGatePrime(Matrix dout, Matrix state, Matrix o) {
        int numberOfSamples = input.getDepthCount();
        Matrix ones = new Matrix(numberOfSamples, units);
        ones.initialize(1d);

        return dout.elementWiseProduct(tanhActivation.activate(state))
                .elementWiseProduct(o)
                .elementWiseProduct(ones.subtract(o));
    }

    private Matrix handleForgetGatePrime(Matrix dState, Matrix prevState, Matrix f) {
        int numberOfSamples = input.getDepthCount();
        Matrix ones = new Matrix(numberOfSamples, units);
        ones.initialize(1d);

        return dState.elementWiseProduct(prevState)
                .elementWiseProduct(f)
                .elementWiseProduct(ones.subtract(f));
    }

    private Matrix handleInputGatePrime(Matrix dState, Matrix a, Matrix i) {
        int numberOfSamples = input.getDepthCount();
        Matrix ones = new Matrix(numberOfSamples, units);
        ones.initialize(1d);

        return dState.elementWiseProduct(a)
                .elementWiseProduct(i)
                .elementWiseProduct(ones.subtract(i));
    }

    private Matrix handleInputActivationPrime(Matrix dState, Matrix i, Matrix a) {
        int numberOfSamples = input.getDepthCount();
        Matrix ones = new Matrix(numberOfSamples, units);
        ones.initialize(1d);

        return dState.elementWiseProduct(i)
                .elementWiseProduct(ones.subtract(a.square()));
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

        inputActivationUWeight.updateWeights(inputActivationDeltaUWeight, learningRate);
        inputGateUWeight.updateWeights(inputGateDeltaUWeight, learningRate);
        forgetGateUWeight.updateWeights(forgetGateDeltaUWeight, learningRate);
        outputGateUWeight.updateWeights(outputGateDeltaUWeight, learningRate);

        inputActivationWeight.updateWeights(inputActivationDeltaWeight, learningRate);
        inputGateWeight.updateWeights(inputGateDeltaWeight, learningRate);
        forgetGateWeight.updateWeights(forgetGateDeltaWeight, learningRate);
        outputGateWeight.updateWeights(outputGateDeltaWeight, learningRate);

        inputActivationBias.updateBias(inputActivationDeltaBias, learningRate);
        inputGateBias.updateBias(inputGateDeltaBias, learningRate);
        forgetGateBias.updateBias(forgetGateDeltaBias, learningRate);
        outputGateBias.updateBias(outputGateDeltaBias, learningRate);


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
