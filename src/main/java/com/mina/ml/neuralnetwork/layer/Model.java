package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.Optimizer;
import org.javatuples.Pair;

import java.util.List;
import java.util.function.Consumer;

public abstract class Model {

    public abstract void summary(Consumer consumer);

    public abstract void compile(Optimizer optimizer, String loss, String metrics);

    public abstract void fit(List<double[]> xTrain, List<double[]> yTrain, float validationSplit,boolean shuffle, int batchSize, int epochs, Verbosity verbosity);

    public abstract Pair<Double, Double> evaluate(List<double[]> xTest, List<double[]> yTest);

//    save_weights(model_file_path);
//    load_weights(model_file_path);
//    rounded_predictions = model.predict_classes(x_test);
//    predictions = model.predict(x_test);

}
