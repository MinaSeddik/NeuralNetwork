package com.mina.ml.neuralnetwork;

import com.google.common.base.Stopwatch;
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
import java.util.concurrent.TimeUnit;
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

    private List<double[]> x_train;
    private List<double[]> y_train;

    private List<double[]> x_test;
    private List<double[]> y_test;


//    private List<double[]> inputData;
//    private List<double[]> outputLabels;

    private double validation_split = 0.15d;

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

    public void fetchDataSet(double[][] inputData, double[][] labels) {

        // do some validation
        if (inputData.length != labels.length) {
            logger.error("Invalid Data set examples, The size of the input data should match the size of the labels for supervised learning!");
            throw new RuntimeException("Invalid DataSet.");
        }

        logger.debug("start fetching data, Total examples = {}", inputData.length);

        // convert the data set to Array-list
        List<double[]> inputs = twoDArrayToList(inputData);
        List<double[]> outputs = twoDArrayToList(labels);

        fetchDataSet(inputs, outputs);
    }

    public void fetchDataSet(List<double[]> inputData, List<double[]> outputLabels) {

        // do some validation
        if (inputData.size() != outputLabels.size()) {
            logger.error("Invalid Data set examples, The size of the input data should match the size of the labels for supervised learning!");
            throw new RuntimeException("Invalid DataSet.");
        }

        logger.debug("Start fetching data, Total examples = {}", inputData.size());

//        this.inputData = inputData;
//        this.outputLabels = outputLabels;

        int validationStartIndex = (int) (inputData.size() * (1d - validation_split));

        x_train = inputData.subList(0, validationStartIndex);
        y_train = outputLabels.subList(0, validationStartIndex);

        x_test = inputData.subList(validationStartIndex, inputData.size());
        y_test = outputLabels.subList(validationStartIndex, outputLabels.size());

//        System.out.println("Validation x_train size = " + x_train.size());
//        System.out.println("Validation y_train size = " + y_train.size());
//
//        System.out.println("Validation x_test size = " + x_test.size());
//        System.out.println("Validation y_test size = " + y_test.size());
    }

    public void train() {
        // make sure that you received the data
        if (Objects.isNull(x_train) || Objects.isNull(y_train)) {
            logger.error("There is No Data set fetched for training, You should fetch the data before start training.");
            throw new RuntimeException("There is No DataSet fetched!");
        }

        logger.debug("Partition the data into batches of size = {}", batchSize);
        List<List<double[]>> inputBatches = Lists.partition(x_train, batchSize);
        List<List<double[]>> outputBatches = Lists.partition(y_train, batchSize);

        int epochNumber = 1;
        double meanErrorPerEpoch;
        List<Double> errorsPerEpoch = new ArrayList<>();
        Stopwatch stopwatch;
        do {
//            logger.info("Number of batches = {}", inputBatches.size());
//            logger.debug("Number of batches = {}", inputBatches.size());
//            Stopwatch sw = Stopwatch.createStarted();


            stopwatch = Stopwatch.createStarted();
            String log = "Epoch " + epochNumber + "/" + maxEpoch;
            System.out.println(log);
//            logger.info(log);
            for (int batch = 0; batch < inputBatches.size(); batch++) {
//logger.info("HERE 1");
//                logger.debug("Prepare Input batch Id = [{}]", batch);
//                logger.info("Prepare Input batch Id = [{}]", batch);


                List<double[]> inputBatch = inputBatches.get(batch);
                double[][] in = new double[inputBatch.size()][];
                in = inputBatch.toArray(in);

//sw.stop();
//long t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 2 - timeElapsed = " + t);
//sw.reset();
//                logger.debug("Prepare Output batch Id = [{}]", batch);
                List<double[]> outputBatch = outputBatches.get(batch);
                double[][] labels = new double[outputBatch.size()][];
                labels = outputBatch.toArray(labels);
//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 3 - timeElapsed = " + t);
//sw.reset();
//                logger.debug("Start Forward propagation for batch Id = [{}]", batch);
                double[][] output = inputLayer.input(in).forwardPropagation();
//                logger.debug("Finish Forward propagation for batch Id = [{}]", batch);
//logger.info("HERE 4");


//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 4 - timeElapsed = " + t);
//sw.reset();


                double meanError = lossFunction.reducedMeanError(labels, output);
//System.out.println("*-*-*-* " + meanError);
//logger.info("HERE 5");
//                logger.debug("Mean Error for batch Id [{}] = {}", batch, meanError);
//                logger.info("Mean Error for batch Id [{}] = {}", batch, meanError);

//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 5 - timeElapsed = " + t);
//sw.reset();


                errorsPerEpoch.add(meanError);
//logger.info("HERE 6");
                // back propagation
//                logger.debug("Start Back propagation for batch Id = [{}]", batch);
                double[][] costOutputPrime = lossFunction.errorOutputPrime(labels, output,
                        outputLayer.getActivationFunction());
//logger.info("HERE 7");


//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 6 - timeElapsed = " + t);
//sw.reset();


                outputLayer.backPropagation(costOutputPrime);
//                logger.debug("Finish Back propagation for batch Id = [{}]", batch);
//logger.info("HERE 8");


//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 7 - timeElapsed = " + t);
//sw.reset();


                // update Weight for all neural nodes
//                logger.debug("Update Weights for batch Id = [{}]", batch);
                inputLayer.updateWeights();
//logger.info("HERE 9");

//sw.stop();
//t = sw.elapsed(TimeUnit.MICROSECONDS);
//logger.info("HERE 8 - timeElapsed = " + t);
//sw.reset();
//
//System.exit(0);
            }


            meanErrorPerEpoch = errorsPerEpoch.stream()
                    .collect(Collectors.summarizingDouble(Double::doubleValue))
                    .getAverage();


//            String report = String.format("Epoch [ %7d ] Mean Error = %.9f", epochNumber, meanErrorPerEpoch);
//            System.out.println(report);
//            logger.debug(report);

//            System.out.println("before: " + errorsPerEpoch.size());
            errorsPerEpoch.clear();
//            System.out.println("after: " + errorsPerEpoch.size());
//            logger.debug("errorsPerEpoch List cleared, size = {}", errorsPerEpoch.size());


            /* validation */

            double[][] x_test_in = new double[x_test.size()][];
            double[][] y_test_in = new double[y_test.size()][];
            double[][] validation_output = inputLayer.input(x_test.toArray(x_test_in))
                    .forwardPropagation();
            double meanError = lossFunction.reducedMeanError(validation_output, y_test.toArray(y_test_in));

            //-----------------------------------------

            stopwatch.stop();
            long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);


//            logger.info(" - " + timeElapsed + "s - loss: 0.3798 - acc: 0.8637 - val_loss: 0.4116 - val_acc: 0.8400");
//            String.format("Epoch [ %7d ] Mean Error = %.9f", epochNumber, meanErrorPerEpoch)
            log = String.format(" - %ds - loss: %.4f - acc: 0.8637 - val_loss: %.4f - val_acc: 0.8400",
                    timeElapsed, meanErrorPerEpoch, meanError);
            System.out.println(log);
//            logger.info(log);

            epochNumber++;

        } while (epochNumber <= maxEpoch);// && Math.abs(meanErrorPerEpoch - EPSILON) > EPSILON);

    }

    private List<double[]> twoDArrayToList(double[][] twoDArray) {
        List<double[]> list = new ArrayList<>();
        for (double[] array : twoDArray) {
            list.add(array);
        }
        return list;
    }

}
