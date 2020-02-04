package com.mina.ml.neuralnetwork.activationfunction;

import com.mina.ml.neuralnetwork.util.D4Matrix;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by menai on 2019-01-31.
 */
public class SoftMax extends ActivationFunction {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(SoftMax.class);

    @Override
    public double[][] activate(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            activate(matrix, result, i);
        }

        return result;
    }

    @Override
    public D4Matrix activate(D4Matrix matrix) {
        D4Matrix output = new D4Matrix(matrix.getDimensionCount(), matrix.getDepthCount(),
                matrix.getRowCount(), matrix.getColumnCount());

        for (int dim = 0; dim < matrix.getDimensionCount(); dim++) {
            for (int depth = 0; depth < matrix.getDepthCount(); depth++) {
                Matrix temp = matrix.getSubMatrix(dim, depth);
                Matrix activated = activate(temp);
                output.setMatrix(dim, depth, activated);
            }
        }

        return output;
    }

    /* new implementation */
    @Override
    public Matrix activate(Matrix matrix) {

        // Normalize the matrix
        // Reference: https://stats.stackexchange.com/questions/304758/softmax-overflow
        Matrix normalizedMatrix = normalize(matrix);

        List<Vector> list = normalizedMatrix.asVectors();
        BigDecimal[] sumPerRow = new BigDecimal[list.size()];
        IntStream.range(0, list.size())
                .parallel()
                .forEach(i -> sumPerRow[i] = sum(list.get(i)));

        Function function = p -> calcSoftMax((Pair<Double, BigDecimal>) p);
        return normalizedMatrix.apply(sumPerRow, function);
    }

    private Matrix normalize(Matrix matrix) {

        List<Vector> list = matrix.asVectors();

        double[] maximums = new double[list.size()];
        IntStream.range(0, list.size())
                .parallel()
                .forEach(i -> maximums[i] = list.get(i).argMax());

        // Normalize the matrix
        return matrix.subtract(new Vector(maximums));
    }

    private double calcSoftMax(Pair<Double, BigDecimal> pair){
        double val = pair.getValue0();
        BigDecimal total = pair.getValue1();

        BigDecimal value = new BigDecimal(activate(val));
        value = value.divide(total, 12, RoundingMode.HALF_UP);
        return value.doubleValue();
    }


    private BigDecimal sum(Vector vector){
        BigDecimal total = new BigDecimal(0d);

        List<BigDecimal> expTotal = Arrays.stream(vector.asArray())
                .mapToObj(val -> new BigDecimal(Math.exp(val)))
                .collect(Collectors.toList());

        for (BigDecimal v : expTotal){
            total = total.add(v);
        }

        return total;
    }

    private void activate(double[][] matrix, double[][] result, int row) {
        BigDecimal sum = new BigDecimal(0d);

        for (int col = 0; col < matrix[0].length; col++) {
            sum = sum.add(new BigDecimal(Math.exp(matrix[row][col])));
        }

        for (int col = 0; col < matrix[0].length; col++) {
            BigDecimal value = new BigDecimal(activate(matrix[row][col]));
            value = value.divide(sum, 12, RoundingMode.HALF_UP);
            result[row][col] = value.doubleValue();
//            result[row][col] = activate(matrix[row][col]) / sum;
        }
    }

    @Override
    public double activate(double value) {
        return Math.exp(value);
    }

    @Override
    public double activatePrime(double value) {
        // un-defined function
        return 1d;
    }

}
