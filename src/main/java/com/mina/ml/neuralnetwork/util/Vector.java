package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class Vector extends Tensor {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Vector.class);

    //    private double[] vector;
    private double[] collection;

    public Vector(int count) {
        collection = new double[count];
    }

    public Vector(double[] list) {
        this(list.length);
        System.arraycopy(list, 0, collection, 0, list.length);
    }

    public Matrix toMatrix() {
        Matrix matrix = new Matrix(1, collection.length);
        matrix.setRow(0, this);
        return matrix;
    }

    public Vector apply(List<Vector> v1, List<Vector> v2, Function<Pair<Vector, Vector>, Double> function) {
        assert v1.size() == v2.size();
        double[] result = new double[v1.size()];
        parallelizeOperation((start, end) -> apply(result, v1, v2, function, start, end));

        return new Vector(result);
    }

    public int size() {
        return collection.length;
    }

    public double getElement(int i) {
        return collection[i];
    }

    public double[] asArray() {
        return collection;
    }

    public double argMax() {
        return Arrays.stream(collection).max().getAsDouble();
    }

    public int argMaxIndex() {
        int maxIndex = 0;
        for (int i = 1; i < collection.length; i++) {
            if (collection[i] > collection[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void apply(double[] result, List<Vector> v1, List<Vector> v2,
                       Function<Pair<Vector, Vector>, Double> function, int startIndex, int endIndex) {

        for (int i = startIndex; i < endIndex; i++) {
            result[i] = function.apply(new Pair<>(v1.get(i), v2.get(i)));
        }
    }

    @Override
    public int getSize() {
        return collection.length;
    }

    @Override
    public String shape() {
        return String.format("(%d)", collection.length);
    }

    @Override
    public boolean sameShape(Tensor tensor) {
        Vector vec = (Vector) tensor;
        return getSize() == vec.getSize();
    }

    public void divide(double val) {
        for (int i = 0; i < collection.length; i++) {
            collection[i]/= val;
        }
    }
}
