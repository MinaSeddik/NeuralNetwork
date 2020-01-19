package com.mina.ml.neuralnetwork.activationfunction;

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

    private final static Logger logger = LoggerFactory.getLogger(SoftMax.class);

    @Override
    public double[][] activate(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            activate(matrix, result, i);
        }

        return result;
    }

    /* new implementation */
    @Override
    public Matrix activate(Matrix matrix) {
        List<Vector> list = matrix.asVectors();
        BigDecimal[] sumPerRow = new BigDecimal[list.size()];
        IntStream.range(0, list.size())
                .parallel()
                .forEach(i -> sumPerRow[i] = sum(list.get(i)));

        Function function = p -> calcSoftMax((Pair<Double, BigDecimal>) p);
        return matrix.apply(sumPerRow, function);
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
