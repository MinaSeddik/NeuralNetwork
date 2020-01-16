package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Matrix extends CollectionParallelizer<double[][]>{

    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

//    protected double[][] matrix;

    public Matrix(int rows, int columns) {
        collection = new double[rows][columns];
    }

    public Matrix(double[][] matrix) {
        collection = matrix;
    }

    public Matrix(List<double[]> list) {
        collection = new double[list.size()][list.get(0).length];
        parallelizeOperation((start, end) -> list2Array(list, start, end));
    }

    public Matrix initialize(double value) {
        parallelizeOperation((start, end) -> initialize(value, start, end));

        return this;
    }

    public Matrix reset() {
        return initialize(0d);
    }

    public Matrix dot(Matrix mat) {
        assert collection[0].length == mat.collection.length;

        double[][] result = new double[getRowCount()][mat.getColumnCount()];
        parallelizeOperation((start, end) -> dot(result, mat.getMatrix(), start, end));
        collection = result;

        return this;
    }

    public Matrix elementWiseProduct(double[][] mat) {
        double[][] result = new double[collection.length][mat[0].length];
        parallelizeOperation((start, end) -> elementWiseProduct(result, mat, start, end));
        collection = result;

        return this;
    }

    public Matrix add(Matrix mat) {
        parallelizeOperation((start, end) -> add(mat.getMatrix(), start, end));

        return this;
    }

    public Matrix divide(double value) {
        parallelizeOperation((start, end) -> divide(value, start, end));

        return this;
    }

    public Matrix addColumn(double value) {
        double[][] result = new double[collection.length][collection[0].length + 1];
        parallelizeOperation((start, end) -> addColumn(result, value, start, end));
        collection = result;

        return this;
    }

    public Matrix transpose() {
        double[][] result = new double[collection[0].length][collection.length];
        parallelizeOperation((start, end) -> transpose(result, start, end));
        collection = result;

        return this;
    }

    public Matrix removeFirstColumn() {
        double[][] result = new double[collection.length][collection[0].length - 1];
        parallelizeOperation((start, end) -> removeFirstColumn(result, start, end));
        collection = result;

        return this;
    }

    public Matrix apply(Function<Double, Double> function) {
        parallelizeOperation((start, end) -> apply(function, start, end));

        return this;
    }

    public Matrix clone() {
        double[][] copy = Arrays.stream(collection).map(double[]::clone).toArray(double[][]::new);
        return new Matrix(copy);
    }

    public double[][] getMatrix() {
        return collection;
    }

    public int getRowCount() {
        return collection.length;
    }

    public int getColumnCount() {
        return collection[0].length;
    }

    public Vector getRowAsVector(int index) {
        return new Vector(collection[index]);
    }

    public List<Vector> asVectors() {
        return Arrays.stream(collection).parallel().map(r -> new Vector(r)).collect(Collectors.toList());
    }

    public List<Vector> getRows() {
        List<Vector> vectors = new ArrayList<>();
        for (int i = 0; i < collection.length; i++) {
            vectors.add(getRowAsVector(i));
        }
        return vectors;
    }

    public void setRow(int index, Vector vector) {
        for (int i = 0; i < vector.size(); i++) {
            collection[index][i] = vector.getElement(i);
        }
    }

    private void initialize(double val, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] = val;
            }
        }
    }

    private void dot(double[][] result, double[][] mat, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                for (int k = 0; k < mat.length; k++) {
                    result[i][j] += collection[i][k] * mat[k][j];
                }
            }
        }
    }

    private void elementWiseProduct(double[][] result, double[][] mat, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] * mat[i][j];
            }
        }
    }

    private void add(double[][] mat, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] += mat[i][j];
            }
        }
    }

    private void divide(double value, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] /= value;
            }
        }
    }

    private void addColumn(double[][] result, double value, int startIndex, int endIndex) {
        for (int row = startIndex; row < endIndex; row++) {
            result[row][0] = value;
            for (int i = 1, j = 0; i < result[0].length && j < collection[0].length; i++, j++) {
                result[row][i] = collection[row][j];
            }
        }
    }

    private void transpose(double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[j][i] = collection[i][j];
            }
        }
    }

    private void removeFirstColumn(double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 1; j < collection[0].length; j++) {
                result[i][j - 1] = collection[i][j];
            }
        }
    }

    private void apply(Function<Double, Double> function, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] = function.apply(collection[i][j]);
            }
        }
    }

    private void list2Array(List<double[]> list, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                collection[i][j] = list.get(i)[j];
            }
        }
    }

    public String shape() {
        return String.format("(%d, %d)", collection.length, collection[0].length);
    }

    public boolean sameShape(Matrix mat){
        return getRowCount() == mat.getRowCount() && getColumnCount() == mat.getColumnCount();
    }

    @Override
    public int getSize(){
        return getRowCount();
    }

    public void debugMatrix(String label) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < collection.length; x++) {
            for (double y : collection[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < collection.length - 1 ? "\n" : "");
        }
//        logger.debug(matrixAsString.toString());
        System.out.println(matrixAsString);
    }

    public void printMatrix(String label) {
        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        for (int x = 0; x < collection.length; x++) {
            for (double y : collection[x]) {
                matrixAsString.append(String.format("%.2f ", y));
            }
            matrixAsString.append(x < collection.length - 1 ? "\n" : "");
        }
        logger.info(matrixAsString.toString());
    }

}
