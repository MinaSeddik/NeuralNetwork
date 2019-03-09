package com.mina.ml.neuralnetwork.factory;

import com.mina.ml.neuralnetwork.activationfunction.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collection;

import static com.mina.ml.neuralnetwork.Constants.*;

/**
 * Created by menai on 2019-02-01.
 */
public class ActivationFunctionFactory {

    private final static Logger logger = LoggerFactory.getLogger(ActivationFunctionFactory.class);

    private Collection<String> activationFunctions = Arrays.asList(
            RELU_ACTIVATION_FUNCTION,
            SIGMOID_ACTIVATION_FUNCTION,
            TANSH_ACTIVATION_FUNCTION,
            SOFT_MAX_ACTIVATION_FUNCTION
    );

    public ActivationFunction createActivationFunction(String activationFunctionName) throws Exception {

        if (!activationFunctions.contains(activationFunctionName.toLowerCase())) {
            logger.error("Invalid Activation Function Name '" + activationFunctionName + "'");
            throw new Exception("Invalid Activation Function Name '" + activationFunctionName + "'");
        }

        ActivationFunction activationFunction = null;
        switch (activationFunctionName.toLowerCase()) {
            case RELU_ACTIVATION_FUNCTION:
                activationFunction = new Relu();
                break;
            case SIGMOID_ACTIVATION_FUNCTION:
                activationFunction = new Sigmoid();
                break;
            case TANSH_ACTIVATION_FUNCTION:
                activationFunction = new Tansh();
                break;
            case SOFT_MAX_ACTIVATION_FUNCTION:
                activationFunction = new SoftMax();
                break;
        }

        return activationFunction;
    }
}
