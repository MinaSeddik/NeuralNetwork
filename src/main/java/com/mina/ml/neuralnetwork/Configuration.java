package com.mina.ml.neuralnetwork;


import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collection;
import java.util.Objects;
import java.util.Properties;

import static com.mina.ml.neuralnetwork.Constants.*;

/**
 * Created by menai on 2019-01-31.
 */
public class Configuration {

    private final static Logger logger = LoggerFactory.getLogger(Configuration.class);

    private Properties properties;

    private Collection<String> activationFunctions = Arrays.asList(
            RELU_ACTIVATION_FUNCTION,
            SIGMOID_ACTIVATION_FUNCTION,
            TANSH_ACTIVATION_FUNCTION,
            SOFT_MAX_ACTIVATION_FUNCTION
    );

    private Collection<String> lossFunctions = Arrays.asList(
            MEAN_SQUARED_ERROR_LOSS_FUNCTION,
            CROSS_ENTROPY_LOSS_FUNCTION
    );

    public Configuration(Properties properties) {
        this.properties = properties;
    }

    public void validateConfiguration() throws Exception {
        validateFeatures(properties);
        validateOutput(properties);
        validateHiddenLayers(properties);
        validateLearningRate(properties);
        validateLossFunction(properties);
        validateNumberOfBatches(properties);
    }

    private void validateFeatures(Properties properties) throws Exception {

        int numOfFeatures = (Integer) properties.get(Constants.NUMBER_OF_FEATURES);
//        if (StringUtils.isEmpty(numOfFeatures)) {
//            throw new Exception("Missed Configuration Property: " + Constants.NUMBER_OF_FEATURES);
//        }

//        if (!NumberUtils.isNumber(numOfFeatures)) {
//            throw new Exception("Invalid Parameter Value: " + Constants.NUMBER_OF_FEATURES);
//        }
    }

    private void validateOutput(Properties properties) throws Exception {

        int numOfOutputs = (Integer) properties.get(Constants.NUMBER_OF_OUTPUT_NODES);
//        if (StringUtils.isEmpty(numOfOutputs)) {
//            throw new Exception("Missed Configuration Property: " + Constants.NUMBER_OF_OUTPUT_NODES);
//        }
//
//        if (!NumberUtils.isNumber(numOfOutputs)) {
//            throw new Exception("Invalid Parameter Value: " + Constants.NUMBER_OF_OUTPUT_NODES);
//        }

        // validate activation function
        String activationFunction = (String) properties.get(Constants.OUTPUT_ACTIVATION_FUNCTION);
        if (StringUtils.isEmpty(activationFunction)) {
            throw new Exception("Missed Configuration Property: " + Constants.OUTPUT_ACTIVATION_FUNCTION);
        }

        if (!activationFunctions.contains(activationFunction.toLowerCase())) {
            throw new Exception("UnSupported Activation Function: " + activationFunction);
        }

    }

    private void validateHiddenLayers(Properties properties) throws Exception {

        Integer numOfHiddenLayers = (Integer) properties.get(Constants.NUMBER_OF_HIDDEN_LAYERS);
        if (!Objects.isNull(numOfHiddenLayers)) {

            // make sure that it is a numerical value
//            if (!NumberUtils.isNumber(numOfHiddenLayers)) {
//                throw new Exception("Invalid Parameter Value: " + Constants.NUMBER_OF_HIDDEN_LAYERS);
//            }

            // make sure that the number of nodes and activation function is defined for each layer
//            int hiddenLayersCount = Integer.parseInt(numOfHiddenLayers);
            for (int i = 1; i <= numOfHiddenLayers; i++) {
                int hiddenLayerNodesCount = (Integer) properties.get(
                        Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", Integer.toString(i)));
//                if (StringUtils.isEmpty(hiddenLayerNodesCount)) {
//                    throw new Exception("Missed Configuration Property: " + Constants.NUMBER_OF_HIDDEN_LAYER_NODES + " for the layer " + i);
//                }
                // make sure that it is a numerical value
//                if (!NumberUtils.isNumber(hiddenLayerNodesCount)) {
//                    throw new Exception("Invalid Parameter Value: " + Constants.NUMBER_OF_HIDDEN_LAYER_NODES + " for the layer " + i);
//                }


                String hiddenLayerActivationFunction = (String) properties.get(
                        Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", Integer.toString(i)));
                if (StringUtils.isEmpty(hiddenLayerActivationFunction)) {
                    throw new Exception("Missed Configuration Property: " + Constants.HIDDEN_ACTIVATION_FUNCTION + " for the layer " + i);
                }
                if (!activationFunctions.contains(hiddenLayerActivationFunction.toLowerCase())) {
                    throw new Exception("UnSupported Activation Function: " + hiddenLayerActivationFunction + " for the layer " + i);
                }

            }

        }

    }

    private void validateLearningRate(Properties properties) throws Exception {
        Double learningRate = (Double) properties.get(Constants.LEARNING_RATE);
        String learningRateMechanism = properties.getProperty(Constants.LEARNING_RATE_MECHANISM);

        if (Objects.isNull(learningRate) && StringUtils.isEmpty(learningRateMechanism)) {
            throw new Exception("Invalid Parameter Value: " + Constants.LEARNING_RATE + " or " + Constants.LEARNING_RATE_MECHANISM);
        }

    }

    private void validateLossFunction(Properties properties) throws Exception {
        String lossFunction = (String) properties.get(Constants.LOSS_FUNCTION);
        if (StringUtils.isEmpty(lossFunction)) {
            throw new Exception("Missed Configuration Property: " + Constants.LOSS_FUNCTION);
        }

        if (!lossFunctions.contains(lossFunction)) {
            throw new Exception("UnSupported Loss Function: " + lossFunction);
        }

    }

    private void validateNumberOfBatches(Properties properties) throws Exception {
        int batchSize = (Integer) properties.get(Constants.BATCH_SIZE);
//        if (StringUtils.isEmpty(batchSize)) {
//            throw new Exception("Missed Configuration Property: " + Constants.BATCH_SIZE);
//        }
//
//        if (!NumberUtils.isNumber(batchSize)) {
//            throw new Exception("Invalid Parameter Value: " + Constants.BATCH_SIZE);
//        }
    }


    public int getNumOfFeatures() {
        return (Integer) properties.get(Constants.NUMBER_OF_FEATURES);
    }

    public int getNumOfHiddenLayers() {
        Integer numOfHiddenLayers = (Integer) properties.get(Constants.NUMBER_OF_HIDDEN_LAYERS);
        return Objects.isNull(numOfHiddenLayers) ? 0 : numOfHiddenLayers;
    }

    public int getNumOfNodes(int hiddenLayerNum) {
        int numOfNodes = (int) properties.get(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}",
                Integer.toString(hiddenLayerNum)));

        return numOfNodes;
    }

    public String getActivationFunction(int hiddenLayerNum) {
        String activationFunctionName = (String) properties.get(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}",
                Integer.toString(hiddenLayerNum)));

        return activationFunctionName;
    }

    public int getNumberOfOutputNodes() {
        return (Integer) properties.get(Constants.NUMBER_OF_OUTPUT_NODES);
    }

    public String getOutputActivationFunction() {
        return (String) properties.get(Constants.OUTPUT_ACTIVATION_FUNCTION);
    }

    public int getBatchSize(int defaultValue) {
        Integer batchSize = (Integer) properties.get(Constants.BATCH_SIZE);
        return Objects.isNull(batchSize) ? defaultValue : batchSize;
    }

    public int getMaxEpoch(int defaultValue) {
        Integer maxEpoch = (Integer) properties.get(Constants.MAX_EPOCH);
        return Objects.isNull(maxEpoch) ? defaultValue : maxEpoch;
    }

    public String getLossFunction() {
        return (String) properties.get(Constants.LOSS_FUNCTION);
    }

    public double getLearningRate() {
        return (Double) properties.get(Constants.LEARNING_RATE);
    }
}
