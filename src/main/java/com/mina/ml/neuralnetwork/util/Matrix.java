package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class Matrix extends Tensor {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

    protected double[][] collection;

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

        return new Matrix(result);
    }

    public Matrix elementWiseProduct(Matrix mat) {
        double[][] result = new double[collection.length][mat.collection[0].length];
        parallelizeOperation((start, end) -> elementWiseProduct(result, mat.collection, start, end));

        return new Matrix(result);
    }

    public Matrix add(Matrix mat) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> add(result, mat.getMatrix(), start, end));

        return new Matrix(result);
    }

    public Matrix subtract(Matrix mat) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> subtract(result, mat.getMatrix(), start, end));

        return new Matrix(result);
    }

    public double sumAllElements() {
        double[] vec = new double[collection.length];
        parallelizeOperation((start, end) -> sumAllElements(vec, start, end));

        return Arrays.stream(vec).sum();
    }

    public Matrix divide(double value) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> divide(result, value, start, end));

        return new Matrix(result);
    }

    public Matrix subtract(Vector vector) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> subtract(result, vector.asArray(), start, end));

        return new Matrix(result);
    }

    public Matrix addColumn(double value) {
        double[][] result = new double[collection.length][collection[0].length + 1];
        parallelizeOperation((start, end) -> addColumn(result, value, start, end));

        return new Matrix(result);
    }

    public Matrix transpose() {
        double[][] result = new double[collection[0].length][collection.length];
        parallelizeOperation((start, end) -> transpose(result, start, end));

        return new Matrix(result);
    }

    public Matrix square() {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> square(result, start, end));

        return new Matrix(result);
    }

    public Matrix removeFirstColumn() {
        double[][] result = new double[collection.length][collection[0].length - 1];
        parallelizeOperation((start, end) -> removeFirstColumn(result, start, end));

        return new Matrix(result);
    }

    public Matrix duplicateRow(int n) {
        assert collection.length == 1;

        double[][] result = new double[n][collection[0].length];
        parallelizeOperation((start, end) -> duplicateRow(result, start, end));

        return new Matrix(result);
    }

    public Matrix apply(Function<Double, Double> function) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> apply(result, function, start, end));

        return new Matrix(result);
    }

    public Matrix apply(BigDecimal[] totals, Function<Pair<Double, BigDecimal>, Double> function) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> apply(result, totals, function, start, end));

        return new Matrix(result);
    }

    public Matrix apply(Matrix mat1, Matrix mat2, Function<Pair<Double, Double>, Double> function) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> apply(result, mat1, mat2, function, start, end));

        return new Matrix(result);
    }

    public Matrix addMatrices(List<Matrix> deltas) {
        double[][] result = new double[collection.length][collection[0].length];
        parallelizeOperation((start, end) -> add(result, deltas, start, end));

        return new Matrix(result);
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

    private void add(double[][] result, double[][] mat, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] + mat[i][j];
            }
        }
    }

    private void subtract(double[][] result, double[][] mat, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] - mat[i][j];
            }
        }
    }

    private void sumAllElements(double[] vec, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[i].length; j++) {
                vec[i] += collection[i][j];
            }
        }
    }

    private void add(double[][] result, List<Matrix> deltas, int startIndex, int endIndex) {
        for (Matrix mat : deltas) {
            for (int i = startIndex; i < endIndex; i++) {
                for (int j = 0; j < collection[0].length; j++) {
                    result[i][j] = collection[i][j] + mat.getMatrix()[i][j];
                }
            }
        }
    }

    private void divide(double[][] result, double value, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] / value;
            }
        }
    }

    private void square(double[][] result, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] * collection[i][j];
            }
        }
    }

    private void subtract(double[][] result, double[] values, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = collection[i][j] - values[i];
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

    private void duplicateRow(double[][] result, int startIndex, int endIndex) {
        for (int row = startIndex; row < endIndex; row++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[row][j] = collection[0][j];
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

    private void apply(double[][] result, Function<Double, Double> function, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = function.apply(collection[i][j]);
            }
        }
    }

    private void apply(double[][] result, BigDecimal[] totals, Function<Pair<Double, BigDecimal>, Double> function, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = function.apply(new Pair<Double, BigDecimal>(collection[i][j], totals[i]));
            }
        }
    }

    private void apply(double[][] result, Matrix mat1, Matrix mat2, Function<Pair<Double, Double>, Double> function, int startIndex, int endIndex) {
        for (int i = startIndex; i < endIndex; i++) {
            for (int j = 0; j < collection[0].length; j++) {
                result[i][j] = function.apply(new Pair<>(mat1.collection[i][j], mat2.collection[i][j]));
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

    @Override
    public String shape() {
        return String.format("(%d, %d)", collection.length, collection[0].length);
    }

    @Override
    public boolean sameShape(Tensor tensor) {
        Matrix mat = (Matrix) tensor;
        return getRowCount() == mat.getRowCount() && getColumnCount() == mat.getColumnCount();
    }

    @Override
    public int getSize() {
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


    public Matrix buildMatrix(Vector v1, Vector v2, Vector v3, Vector v4) {
        collection[0] = v1.asArray().clone();
        collection[1] = v2.asArray().clone();
        collection[2] = v3.asArray().clone();
        collection[3] = v4.asArray().clone();

        return this;
    }

    Stream<double[]> stream() {
        return Stream.of(collection[0], collection[1], collection[3]);
    }


}
