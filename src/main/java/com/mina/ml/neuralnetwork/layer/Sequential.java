package com.mina.ml.neuralnetwork.layer;

import com.google.common.base.Stopwatch;
import com.mina.ml.neuralnetwork.factory.LossFunctionFactory;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
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

        Stopwatch stopwatch;
        Layerrr inputLayer = layers.get(0);
        Splitter<double[]> splitter = new Splitter(x, y, shuffle);
        for (int epoch = 1; epoch <= epochs; epoch++) {

            // Handle verbosity
            stopwatch = Stopwatch.createStarted();
            String log = "Epoch " + epoch + "/" + epochs;
            System.out.println(log);

            splitter.reset();
            Quartet<List<double[]>, List<double[]>, List<double[]>, List<double[]>> dataset = splitter.split(validationSplit);
            List<double[]> xTrain = dataset.getValue0();
            List<double[]> yTrain = dataset.getValue1();
            List<double[]> xTest = dataset.getValue2();
            List<double[]> yTest = dataset.getValue3();

            Partitioner<double[]> partitioner = new Partitioner<>(xTrain, yTrain, batchSize);
            while (partitioner.hasNext()) {

                Pair<List<double[]>, List<double[]>> batch = partitioner.getNext();
                Matrix xBatch = new Matrix(batch.getValue0());
                Matrix yBatch = new Matrix(batch.getValue1());


                Matrix yPrimeBatch = inputLayer.forwardPropagation(xBatch);

                double meanError = lossFunction.reducedMeanError2(yBatch, yPrimeBatch);

            }

            stopwatch.stop();
            long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
            System.out.println(timeElapsed + "s");
            System.exit(0);

        }
    }
}
