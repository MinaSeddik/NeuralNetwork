package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.Optimizer;

import java.util.List;
import java.util.function.Consumer;

public abstract class Model {

    public abstract void summary(Consumer consumer);

    public abstract void compile(Optimizer optimizer, String loss, String metrics);

    public abstract void fit(List<double[]> x, List<double[]> y, float validationSplit,boolean shuffle, int batchSize, int epochs, Verbosity verbosity);

}
