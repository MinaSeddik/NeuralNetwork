package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.javatuples.Triplet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class Vector extends Tensor {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Vector.class);

    protected double[] collection;

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

    public double average() {
        return Arrays.stream(collection).average().getAsDouble();
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

    public Vector divide(double val) {
        double[] result = new double[collection.length];
        for (int i = 0; i < collection.length; i++) {
            result[i] = collection[i] / val;
        }

        return new Vector(result);
    }

    public Matrix reshape(Pair<Integer, Integer> shape) {
        int n = shape.getValue0();
        int m = shape.getValue1();
        assert n * m == collection.length;

        int x = 0;
        double[][] result = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = collection[x++];
            }
        }

        return new Matrix(result);
    }

    public D3Matrix reshape(Triplet<Integer, Integer, Integer> shape) {
        int n = shape.getValue0();
        int m = shape.getValue1();
        int l = shape.getValue2();
        assert n * m * l == collection.length;

        int x = 0;
        double[][][] result = new double[n][m][l];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < l; k++) {
                    result[i][j][k] = collection[x++];
                }
            }
        }

        return new D3Matrix(result);
    }

    public Vector add(Vector vector) {
        double[] result = new double[collection.length];
        for(int i=0;i<collection.length;i++){
            result[i] = collection[i] + vector.collection[i];
        }

        return new Vector(result);
    }
}
