package com.mina.ml.neuralnetwork;

import com.google.common.collect.Lists;
import com.mina.ml.neuralnetwork.factory.LossFunctionFactory;
import com.mina.ml.neuralnetwork.layer.HiddenLayer;
import com.mina.ml.neuralnetwork.layer.InputLayer;
import com.mina.ml.neuralnetwork.layer.Layer;
import com.mina.ml.neuralnetwork.layer.OutputLayer;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Created by menai on 2019-01-31.
 */
public class NeuralNetwork {

    private final static Logger logger = LoggerFactory.getLogger(NeuralNetwork.class);

    private static final double EPSILON = 1E-4;
    private static final int DEFAULT_BATCH_SIZE = 512;
    private static final int DEFAULT_MAX_EPOCH = 1000000;

    private Layer inputLayer;
    private Layer outputLayer;

    private List<float[]> inputData;
    private List<float[]> outputLabels;

    private LossFunction lossFunction;
    private int batchSize;

    private int maxEpoch;

    public NeuralNetwork(Configuration configuration) throws Exception {

        logger.debug("Validate Neural Network Configurations ...");
        configuration.validateConfiguration();
        logger.debug("Configurations Validation ... PASS");

        logger.debug("Initialize the Neural Network ...");
        initializeNetwork(configuration);
        logger.debug("Neural Network initialized!");
    }

    private void initializeNetwork(Configuration configuration) {

        logger.debug("Setup Input Layer with Num of Features = {}", configuration.getNumOfFeatures());
        inputLayer = new InputLayer("Input Layer", configuration.getNumOfFeatures());
        Layer prevLayer = inputLayer;

        for (int i = 1; i <= configuration.getNumOfHiddenLayers(); i++) {

            logger.debug("Setup Hidden Layer [{}] with Num of Inputs = {}, Num Of Neurons = {}, " +
                            "Activation Function = {} and Learning Rate = {}",
                    i, prevLayer.getNumberOfOutputs(), configuration.getNumOfNodes(i),
                    configuration.getActivationFunction(i), configuration.getLearningRate());

            Layer layer = new HiddenLayer("Hidden Layer [" + i + "]", prevLayer.getNumberOfOutputs(),
                    configuration.getNumOfNodes(i),
                    configuration.getActivationFunction(i), configuration.getLearningRate());

            layer.setPreviousLayer(prevLayer);
            prevLayer.setNextLayer(layer);
            prevLayer = layer;
        }

        logger.debug("Setup Output Layer with Num of Inputs = {}, Num Of Neurons = {}, " +
                        "Activation Function = {} and Learning Rate = {}",
                prevLayer.getNumberOfOutputs(), configuration.getNumberOfOutputNodes(),
                configuration.getOutputActivationFunction(), configuration.getLearningRate());

        outputLayer = new OutputLayer("Output Layer", prevLayer.getNumberOfOutputs(), configuration.getNumberOfOutputNodes(),
                configuration.getOutputActivationFunction(), configuration.getLearningRate());

        outputLayer.setPreviousLayer(prevLayer);
        prevLayer.setNextLayer(outputLayer);

        LossFunctionFactory lossFunctionFactory = new LossFunctionFactory();
        try {
            logger.debug("Setup the Loss Function = {}", configuration.getLossFunction());
            lossFunction = lossFunctionFactory.createLossFunction(configuration.getLossFunction());
        } catch (Exception ex) {
            logger.error("Exception: {} occurred {}", ex.getClass(), ex);
        }

        batchSize = configuration.getBatchSize(DEFAULT_BATCH_SIZE);
        logger.debug("Set Batch size = {}", batchSize);

        maxEpoch = configuration.getMaxEpoch(DEFAULT_MAX_EPOCH);
        logger.debug("Set Max Epoch = {}", maxEpoch);
    }

    public void fetchDataSet(float[][] inputData, float[][] labels) {

        // do some validation
        if (inputData.length != labels.length) {
            logger.error("Invalid Data set examples, The size of the input data should match the size of the labels for supervised learning!");
            throw new RuntimeException("Invalid DataSet.");
        }

        logger.debug("start fetching data, Total examples = {}", inputData.length);

        // convert the data set to Array-list
        List<float[]> inputs = twoDArrayToList(inputData);
        List<float[]> outputs = twoDArrayToList(labels);

        fetchDataSet(inputs, outputs);
    }

    public void fetchDataSet(List<float[]> inputData, List<float[]> outputLabels) {

        // do some validation
        if (inputData.size() != outputLabels.size()) {
            logger.error("Invalid Data set examples, The size of the input data should match the size of the labels for supervised learning!");
            throw new RuntimeException("Invalid DataSet.");
        }

        logger.debug("Start fetching data, Total examples = {}", inputData.size());

        this.inputData = inputData;
        this.outputLabels = outputLabels;
    }

    public void train() {
        // make sure that you received the data
        if (Objects.isNull(inputData) || Objects.isNull(outputLabels)) {
            logger.error("There is No Data set fetched for training, You should fetch the data before start training.");
            throw new RuntimeException("There is No DataSet fetched!");
        }

        logger.debug("Partition the data into batches of size = {}", batchSize);
        List<List<float[]>> inputBatches = Lists.partition(inputData, batchSize);
        List<List<float[]>> outputBatches = Lists.partition(outputLabels, batchSize);

        int epochNumber = 1;
        double meanErrorPerEpoch;
        List<Double> errorsPerEpoch = new ArrayList<>();
        do {

            logger.debug("Number of batches = {}", inputBatches.size());
            for (int batch = 0; batch < inputBatches.size(); batch++) {

                logger.debug("Prepare Input batch Id = [{}]", batch);
                List<float[]> inputBatch = inputBatches.get(batch);
                float[][] in = new float[inputBatch.size()][];
                in = inputBatch.toArray(in);

                logger.debug("Prepare Output batch Id = [{}]", batch);
                List<float[]> outputBatch = outputBatches.get(batch);
                float[][] labels = new float[outputBatch.size()][];
                labels = outputBatch.toArray(labels);

                logger.debug("Start Forward propagation for batch Id = [{}]", batch);
                float[][] output = inputLayer.input(in).forwardPropagation();
                logger.debug("Finish Forward propagation for batch Id = [{}]", batch);

                double meanError = lossFunction.reducedMeanError(labels, output);
//System.out.println("*-*-*-* " + meanError);
                logger.debug("Mean Error for batch Id [{}] = {}", batch, meanError);
logger.info("Mean Error for batch Id [{}] = {}", batch, meanError);
//batch=3000000;
//epochNumber=3000000;
                errorsPerEpoch.add(meanError);

                // back propagation
                logger.debug("Start Back propagation for batch Id = [{}]", batch);
                float[][] costOutputPrime = lossFunction.errorOutputPrime(labels, output,
                        outputLayer.getActivationFunction());

                outputLayer.backPropagation(costOutputPrime);
                logger.debug("Finish Back propagation for batch Id = [{}]", batch);

                // update Weight for all neural nodes
                logger.debug("Update Weights for batch Id = [{}]", batch);
                inputLayer.updateWeights();

            }

            meanErrorPerEpoch = errorsPerEpoch.stream()
                    .collect(Collectors.summarizingDouble(Double::doubleValue))
                    .getAverage();

            String report = String.format("Epoch [ %7d ] Mean Error = %.9f", epochNumber, meanErrorPerEpoch);
            System.out.println(report);
            logger.debug(report);

            errorsPerEpoch.clear();
            logger.debug("errorsPerEpoch List cleared, size = {}", errorsPerEpoch.size());

            epochNumber++;

        } while (epochNumber <= maxEpoch && Math.abs(meanErrorPerEpoch - EPSILON) > EPSILON );

    }

    private List<float[]> twoDArrayToList(float[][] twoDArray) {
        List<float[]> list = new ArrayList<>();
        for (float[] array : twoDArray) {
            list.add(array);
        }
        return list;
    }

}
