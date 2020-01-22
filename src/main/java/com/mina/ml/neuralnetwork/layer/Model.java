package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.util.FilesUtil;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.List;
import java.util.function.Consumer;

public abstract class Model implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Model.class);

    public abstract void summary(Consumer consumer);

    public abstract void compile(Optimizer optimizer, String loss, String metrics);

    public abstract void fit(List<double[]> xTrain, List<double[]> yTrain, float validationSplit, boolean shuffle, int batchSize, int epochs, Verbosity verbosity, List<ModelCheckpoint> callbacks);

    public abstract Pair<Double, Double> evaluate(List<double[]> xTest, List<double[]> yTest);

    public abstract void loadWeights(String modelFilePath);

    public abstract List<double[]> predict(List<double[]> x);

    public abstract List<Integer> predictClasses(List<double[]> x);

    public void save(String modelFilePath) {
        FilesUtil.serializeData(modelFilePath, this);
    }

    public static Model load(String modelFilePath) {
        return FilesUtil.deSerializeData(modelFilePath);
    }

}
