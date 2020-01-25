package com.mina.ml.neuralnetwork.layer;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableMap;
import com.mina.ml.neuralnetwork.factory.LossFunctionFactory;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.lossfunction.LossFunction;
import com.mina.ml.neuralnetwork.util.*;
import dnl.utils.text.table.TextTable;
import org.apache.commons.collections4.CollectionUtils;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.NumberFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class Sequential extends Model {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Sequential.class);

    private List<Layerrr> layers = new ArrayList<>();
    private LossFunction lossFunction;

    private Optimizer optimizer;

    public static final String ACCURACY_METRICS = "accuracy";
    private String metrics;

    public Sequential() {
    }

    public Sequential(Layerrr[] array) {
        Arrays.stream(array).forEach(e -> add(e));
    }

    @Override
    public void add(Layerrr layer) {
        Layerrr prev = layers.size() > 0 ? layers.get(layers.size() - 1) : null;
        int lastIndex = layers.size() > 0 ? layers.get(layers.size() - 1).getIndex() + 1 : 1;

        // link up this layer
        if (null != prev) {
            prev.setNext(layer);
            layer.setInputShape(prev.getOutputShape());
        }
        layer.setPrevious(prev);
        layer.setIndex(lastIndex);

        layers.add(layer);
    }

    @Override
    public void summary(Consumer consumer) {
        AtomicInteger totalParameters = new AtomicInteger();
        AtomicInteger trainableParameters = new AtomicInteger();
        AtomicInteger nonTrainableParameters = new AtomicInteger();

        String[] columnNames = {"Layer (type)", "Output Shape", "Param #"};
        Object[][] data = new Object[layers.size()][3];
        for (int i = 0; i < layers.size(); i++) {
            data[i][0] = String.format("%s (%s)", layers.get(i).getName(), layers.get(i).getLayerType());
            data[i][1] = layers.get(i).getOutputShape();
            data[i][2] = layers.get(i).getNumberOfParameter();
            totalParameters.addAndGet(layers.get(i).getNumberOfParameter());
            trainableParameters.addAndGet(layers.get(i).getNumberOfParameter());
        }
        TextTable textTable = new TextTable(columnNames, data);
        textTable.setAddRowNumbering(true);
        textTable.printTable();

//        consumer.accept("_________________________________________________________________");
//        consumer.accept("Layer (type)\t\t\t\tOutput Shape\t\t\t\tParam #");
//        consumer.accept("=================================================================");
//        layers.stream().forEach(layer -> {
//            String layerInfo = String.format("%s (%s)\t\t\t%s\t\t\t\t%d",
//                    layer.getName(),
//                    layer.getLayerType(),
//                    layer.getOutputShape(),
//                    layer.getNumberOfParameter());
//            totalParameters.addAndGet(layer.getNumberOfParameter());
//            trainableParameters.addAndGet(layer.getNumberOfParameter());
//            consumer.accept(layerInfo);
////        });
////        consumer.accept("=================================================================");

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

        this.metrics = ACCURACY_METRICS;
    }

    @Override
    public void fit(List<Object> x, List<double[]> y, float validationSplit, boolean shuffle, int batchSize,
                    int epochs, Verbosity verbosity, List<ModelCheckpoint> callbacks) {

        Stopwatch stopwatch = Stopwatch.createStarted();
        Layerrr inputLayer = layers.get(0);
        Layerrr outputLayer = layers.get(layers.size() - 1);
        Splitter splitter = new Splitter(x, y, shuffle);
        for (int epoch = 1; epoch <= epochs; epoch++) {

            stopwatch.reset().start();
            // Handle verbosity
            String log = "Epoch " + epoch + "/" + epochs;
            System.out.println(log);

            splitter.reset();
            Quartet<List<? extends Object>, List<? extends Object>, List<? extends Object>, List<? extends Object>> dataset = splitter.split(validationSplit);


            List<? extends Object> xTrain = dataset.getValue0();
            List<? extends Object> yTrain = dataset.getValue1();
            List<? extends Object> xTest = dataset.getValue2();
            List<? extends Object> yTest = dataset.getValue3();

            Partitioner partitioner = new Partitioner(xTrain, yTrain, batchSize);
            int batchCount = 0;
            Pair<Double, Double> trainStats = new Pair<>(0d, 0d);
            Pair<Double, Double> batchStats;
            while (partitioner.hasNext()) {

                Pair<List<? extends Object>, List<? extends Object>> batch = partitioner.getNext();
//                Matrix xBatch = new Matrix(batch.getValue0());
//                Matrix yBatch = new Matrix(batch.getValue1());

                Tensor xBatch = Tensor.getTensor(batch.getValue0());
                Tensor yBatch = Tensor.getTensor(batch.getValue1());

                batchCount++;
                batchStats = optimizer.optimize(inputLayer, outputLayer, lossFunction, xBatch, yBatch);

                trainStats = new Pair<>(trainStats.getValue0() + batchStats.getValue0(),
                        trainStats.getValue1() + batchStats.getValue1());
            }

            Pair<Double, Double> testLoss = evaluate(xTest, yTest);

            double loss = trainStats.getValue0() / batchCount;
            double acc = trainStats.getValue1() / batchCount;

            double valLoss = testLoss.getValue0();
            double valAcc = testLoss.getValue1();

            stopwatch.stop();
            long timeElapsed = stopwatch.elapsed(TimeUnit.SECONDS);
            log = String.format(" - %ds - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f",
                    timeElapsed, loss, acc, valLoss, valAcc);
            System.out.println(log);

            // check if any callbacks defined
            if (CollectionUtils.isNotEmpty(callbacks)) {
                Map<String, Object> params = ImmutableMap.of(
                        "epoch", epoch,
                        ModelCheckpoint.TRAIN_LOSS, loss,
                        ModelCheckpoint.TRAIN_ACCURACY, acc,
                        ModelCheckpoint.VALIDATION_LOSS, valLoss,
                        ModelCheckpoint.VALIDATION_ACCURACY, valAcc
                );
                // Handle the first callback only
                callbacks.get(0).handle(params, layers);
            }

        }
    }

    @Override
    public Pair<Double, Double> evaluate(List<? extends Object> data, List<? extends Object> labels) {

        Layerrr inputLayer = layers.get(0);

//        Matrix x = new Matrix(data);
//        Matrix y = new Matrix(labels);

        Tensor x = Tensor.getTensor(data);
        Matrix y = (Matrix) Tensor.getTensor(labels);

        Matrix yPrime = (Matrix) inputLayer.forwardPropagation(x);

        double loss = lossFunction.meanErrorCost(y, yPrime);
        double acc = lossFunction.calculateAccuracy(y, yPrime);

        return new Pair<>(loss, acc);
    }

    @Override
    public void loadWeights(String modelFilePath) {
        Map<Integer, WeightMatrix> weights = FilesUtil.deSerializeData(modelFilePath);
        layers.stream().forEach(l -> l.setWeights(weights.get(l.getIndex())));
    }

    @Override
    public List<double[]> predict(List<double[]> x) {
        Layerrr inputLayer = layers.get(0);

        Matrix input = new Matrix(x);
        Matrix output = (Matrix) inputLayer.forwardPropagation(input);

        return output.asVectors().stream().map(v -> v.asArray()).collect(Collectors.toList());
    }

    @Override
    public List<Integer> predictClasses(List<double[]> x) {
        Layerrr inputLayer = layers.get(0);

        Matrix input = new Matrix(x);
        Matrix output = (Matrix) inputLayer.forwardPropagation(input);

        return output.asVectors().stream().map(v -> v.argMaxIndex()).collect(Collectors.toList());
    }


}
