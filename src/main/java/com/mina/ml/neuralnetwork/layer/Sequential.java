package com.mina.ml.neuralnetwork.layer;

import com.google.common.base.Stopwatch;
import com.mina.ml.neuralnetwork.factory.LossFunctionFactory;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.Partitioner;
import com.mina.ml.neuralnetwork.util.Splitter;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Sequential extends Model {

    private final static Logger logger = LoggerFactory.getLogger(Sequential.class);

    private List<Dense> layers = new ArrayList<>();
    private LossFunction lossFunction;

    private Optimizer optimizer;

    public Sequential(Dense[] array) {
        Arrays.stream(array).forEach(e -> add(e));
    }

    public void add(Dense layer) {
        Dense prev = layers.size() > 0 ? layers.get(layers.size() - 1) : null;
        int lastIndex = layers.size() > 0 ? layers.get(layers.size() - 1).getIndex() + 1 : 1;

        // link up this layer
        if (null != prev) {
            prev.setNextDense(layer);
            layer.setInputParameters(prev.getOutputParameters());
        }
        layer.setPreviousDense(prev);
        layer.setIndex(lastIndex);

        layers.add(layer);
    }

    @Override
    public void summary(Consumer consumer) {
        AtomicInteger totalParameters = new AtomicInteger();
        AtomicInteger trainableParameters = new AtomicInteger();
        AtomicInteger nonTrainableParameters = new AtomicInteger();

        consumer.accept("_________________________________________________________________");
        consumer.accept("Layer (type)\t\t\t\t\tOutput Shape\t\t\tParam #");
        consumer.accept("=================================================================");
        layers.stream().forEach(layer -> {
            String layerInfo = String.format("%s (%s)\t\t\t\t\t(None, %s)\t\t\t\t%d", layer.getName(), layer.getType(),
                    layer.getOutputParameters(), layer.getNumberOfParameter());
            totalParameters.addAndGet(layer.getNumberOfParameter());
            trainableParameters.addAndGet(layer.getNumberOfParameter());
            consumer.accept(layerInfo);
        });
        consumer.accept("=================================================================");
        consumer.accept(String.format("Total params: %s",
                NumberFormat.getNumberInstance(Locale.US).format(totalParameters.get())));
        consumer.accept(String.format("Trainable params: %s",
                NumberFormat.getNumberInstance(Locale.US).format(trainableParameters.get())));
        consumer.accept(String.format("Non-trainable params: %s",
                NumberFormat.getNumberInstance(Locale.US).format(nonTrainableParameters.get())));
        consumer.accept("_________________________________________________________________");
    }

    @Override
    public void compile(Optimizer optimizer, String loss, String metrics) {
        layers.stream().forEach(layer -> layer.buildupLayer());

        this.optimizer = optimizer;

        LossFunctionFactory lossFunctionFactory = new LossFunctionFactory();
        try {
            logger.debug("Setup the Loss Function = {}", loss);
            lossFunction = lossFunctionFactory.createLossFunction(loss);
        } catch (Exception ex) {
            logger.error("{}: {}", ex.getClass(), ex);
            throw new RuntimeException(ex.getMessage());
        }

        // TODO: handle the metrics

    }


    @Override
    public void fit(List<double[]> x, List<double[]> y, float validationSplit, boolean shuffle, int batchSize, int epochs, Verbosity verbosity) {
        Stopwatch stopwatch = Stopwatch.createStarted();
        Layerrr inputLayer = layers.get(0);
        Layerrr outputLayer = layers.get(layers.size()-1);
        Splitter<double[]> splitter = new Splitter(x, y, shuffle);
        for (int epoch = 1; epoch <= epochs; epoch++) {

            stopwatch.reset().start();
            // Handle verbosity
            String log = "Epoch " + epoch + "/" + epochs;
            System.out.println(log);

            splitter.reset();
            Quartet<List<double[]>, List<double[]>, List<double[]>, List<double[]>> dataset = splitter.split(validationSplit);
            List<double[]> xTrain = dataset.getValue0();
            List<double[]> yTrain = dataset.getValue1();
            List<double[]> xTest = dataset.getValue2();
            List<double[]> yTest = dataset.getValue3();

            Partitioner<double[]> partitioner = new Partitioner<>(xTrain, yTrain, batchSize);
            double meanError = 0;
            int batchCount = 0;
            while (partitioner.hasNext()) {

//                Stopwatch s1 = Stopwatch.createStarted();
                Pair<List<double[]>, List<double[]>> batch = partitioner.getNext();
                Matrix xBatch = new Matrix(batch.getValue0());
                Matrix yBatch = new Matrix(batch.getValue1());
                batchCount++;

//                s1.stop().start();
//                long t = stopwatch.elapsed(TimeUnit.SECONDS);
//                System.out.println(String.format("Batch %d = %ds", batchCount, t));

                meanError = optimizer.optimize(inputLayer, outputLayer, lossFunction, xBatch, yBatch);
//
//                s1.stop();
//                long t = s1.elapsed(TimeUnit.SECONDS);
//                System.out.println(String.format("Batch %d = %ds", batchCount, t));

            }

            meanError/= batchCount;
            stopwatch.stop();
            long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
            System.out.println(timeElapsed + "s meanError = " + meanError);

        }
    }

    private void compareWeights(Layerrr nl, Layer ol) {

        // compare input layer
        double[][] n1 = nl.getW();

        ol = ol.nextLayer;
        double[][] o1 = ol.getW();
        System.out.println("first input layer weights comparison: " + MatrixManipulator.compare(n1, o1));

        nl = nl.nextDense;
        ol = ol.nextLayer;
        n1 = nl.getW();
        o1 = ol.getW();
        System.out.println("second input layer weights comparison: " + MatrixManipulator.compare(n1, o1));

        nl = nl.nextDense;
        ol = ol.nextLayer;
        n1 = nl.getW();
        o1 = ol.getW();
        System.out.println("3rd input layer weights comparison: " + MatrixManipulator.compare(n1, o1));

//        System.exit(0);

    }

    private void compareAZ(Layerrr nl, Layer ol) {

        // compare input layer
        double[][] n1 = nl.getA();

        ol = ol.nextLayer;
        double[][] o1 = ol.getA();
        System.out.println("first layer A comparison: " + MatrixManipulator.compare(n1, o1));
        n1 = nl.getZ();
        o1 = ol.getZ();
        System.out.println("first layer Z comparison: " + MatrixManipulator.compare(n1, o1));

        nl = nl.nextDense;
        ol = ol.nextLayer;
        n1 = nl.getA();
        o1 = ol.getA();
        System.out.println("second layer A comparison: " + MatrixManipulator.compare(n1, o1));
        n1 = nl.getZ();
        o1 = ol.getZ();
        System.out.println("second layer Z comparison: " + MatrixManipulator.compare(n1, o1));

        nl = nl.nextDense;
        ol = ol.nextLayer;
        n1 = nl.getA();
        o1 = ol.getA();
        System.out.println("third layer A comparison: " + MatrixManipulator.compare(n1, o1));
        n1 = nl.getZ();
        o1 = ol.getZ();
        System.out.println("third layer Z comparison: " + MatrixManipulator.compare(n1, o1));
        System.exit(0);

//        System.exit(0);

    }

    private Layer inputLayerOld;
    private Layer outputLayer;

    private void setupOld() {

        inputLayerOld = new InputLayer("Input Layer", 28*28);
        Layer prevLayer = inputLayerOld;

        // Hidden Layer 1
        Layer layer = new HiddenLayer("Hidden Layer [" + 1 + "]", prevLayer.getNumberOfOutputs(),
                64, "relu", 0.001);
        layer.setPreviousLayer(prevLayer);
        prevLayer.setNextLayer(layer);
        prevLayer = layer;

        // Hidden Layer 2
        layer = new HiddenLayer("Hidden Layer [" + 2 + "]", prevLayer.getNumberOfOutputs(),
                64, "relu", 0.001);
        layer.setPreviousLayer(prevLayer);
        prevLayer.setNextLayer(layer);
        prevLayer = layer;


        outputLayer = new OutputLayer("Output Layer", prevLayer.getNumberOfOutputs(), 10,
                "softmax", 0.001);

        outputLayer.setPreviousLayer(prevLayer);
        prevLayer.setNextLayer(outputLayer);

        LossFunctionFactory lossFunctionFactory = new LossFunctionFactory();
        try {
            lossFunction = lossFunctionFactory.createLossFunction("CrossEntropyLoss");
        } catch (Exception ex) {
            logger.error("Exception: {} occurred {}", ex.getClass(), ex);
        }

        int batchSize = 128;
        int maxEpoch = 100;
    }
}
