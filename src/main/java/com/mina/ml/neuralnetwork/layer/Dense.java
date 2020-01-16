package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.factory.ActivationFunctionFactory;
import com.mina.ml.neuralnetwork.util.Matrix;
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
    protected ActivationFunction activationFunction;

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

//    @Override
//    public Matrix forwardPropagation(Matrix input){
//
//        Matrix Z = null;
//        switch (networkLayerType){
////            case INPUT:
////                //        input Matrix (Input Layer Dense):
////                //        ---------------------------------
////                //        | x1 x2 x3 x4    ....    xn |
////                //        | x1 x2 x3 x4    ....    xn |
////                //        | .. .. ..       ....    .. |
////                //        | .. .. ..       ....    .. |
////                //        | x1 x2 x3 x4    ....    xn |
////
////                //        output Matrix (Input Layer Dense):
////                //        - Add bais to the date (x0=1)
////                //        ---------------------------------
////                //        | x0 x1 x2 x3 x4    ....    xn |
////                //        | x0 x1 x2 x3 x4    ....    xn |
////                //        | .. .. .. ..       ....    .. |
////                //        | .. .. .. ..       ....    .. |
////                //        | x0 x1 x2 x3 x4    ....    xn |
////
////                //        No Operations (Input Layer Dense):
////
////                System.out.println("Input input = " + input.shape());
////                Z = addBias(input);
////                System.out.println("After adding bias = " + Z.shape());
////                Z = nextDense.forwardPropagation(Z);
////                break;
//            case HIDDEN:
//                input = addBias(input);
//                System.out.println("Hidden, after adding bias input = " + input.shape());
//                System.out.println("Hidden weight = " + weight.shape());
//
//                Matrix dot = input.dot(weight);
//                Z = activationFunction.activate(dot);
//                Z = addBias(Z);
//
//
////                Z = nextDense.forwardPropagation(activationFunction.activate(dot)).addColumn(1d);
//                break;
//            case OUTPUT:
//                System.out.println("Output, input = " + input.shape());
//                System.out.println("Output weight = " + weight.shape());
//                Z = activationFunction.activate(input.dot(weight));
//                break;
//            default:
//                RuntimeException ex = new RuntimeException("InValid Network Dense Type.");
//                logger.error("{}, Exception: {}", ex.getMessage(), ex);
//                throw ex;
//        }
//        return Z;
//
//
//
//
//    }

    @Override
    public Matrix forwardPropagation(Matrix input) {
//        System.out.println("Input shape: " + input.shape());
//        if( Objects.isNull(previousDense) ){
//            // Add Bias, If it is the Input Layer
        input = addBias(input);
//            System.out.println("Input shape (after bias): " + input.shape());
//        }
//        System.out.println("weight shape: " + weight.shape());
        Matrix A = input.dot(weight);
//        System.out.println("A shape: " + A.shape());
        Matrix Z = activationFunction.activate(A);
//        System.out.println("Z shape: " + Z.shape());

        return Objects.isNull(nextDense) ? Z : nextDense.forwardPropagation(Z);
    }

    public Matrix addBias(Matrix matrix) {
        return matrix.addColumn(1d);
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
