package com.mina.examples;

import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.*;
import com.mina.ml.neuralnetwork.util.D3Matrix;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Triplet;
import org.javatuples.Tuple;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class LstmSeries2 {

    public static void main(String[] args) {

        List<double[][]> X = new ArrayList<>();
        List<double[]> Y = new ArrayList<>();

        double[] temp = new double[20];
        IntStream.range(0, 20).forEach(i -> temp[i] = i + 1);
        IntStream.range(1, 21).forEach(i -> Y.add(new double[]{i * 15d}));

        D3Matrix data = new Vector(temp).reshape(new Triplet<>(20, 1, 1));
//        System.out.println("data.shape " + data.shape());
        for(int i=0;i<data.getDepthCount();i++){
            System.out.println(i + " " + data.getSubMatrix(i).shape());
            X.add(data.getSubMatrix(i).getMatrix());
        }

        Tuple inputShape = new Triplet<>(0, 1, 1);

        Model model = new Sequential();
        model.add(new LSTM(100, inputShape, "relu"));
        model.add(new Dense(1, "relu"));

        model.summary(line -> System.out.println(line));

        double learningRate = 0.001;
        Optimizer optimizer = new Optimizer(learningRate);
        model.compile(optimizer, "MeanSquaredError", "");

        model.fit(X, Y, 0.1f, true, 128, 300, Verbosity.VERBOSE, null);

//        Pair<Double, Double> testStats = model.evaluate(xTest, yTest);
//        double test_acc = testStats.getValue1();
//        System.out.println(String.format("Test accuracy: %.2f%%", (test_acc * 100)));

    }
}
