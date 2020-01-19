package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(LossFunction.class);

    public double reducedMeanError(double[][] labels, double[][] output) {
        double[][] costs = errorCost(labels, output);

//        MatrixManipulator.debugMatrix("Reduced Mean Error Matrix:", costs);
//        MatrixManipulator.printMatrix("Reduced Mean Error Matrix:", costs);
        assert (costs[0].length == 1);

        double meanError = 0d;
        for (int i = 0; i < costs.length; i++) {
            meanError += costs[i][0];
        }

//        logger.info("mean error: [inside reducedMeanError] = " + meanError);
//        logger.info("mean error: [costs.length] = " + costs.length);
//        logger.info("mean error: [final] = " + meanError / costs.length);
        return meanError / costs.length;
    }

    public double meanErrorCost(Matrix y, Matrix yPrime) {
        assert y.sameShape(yPrime);

        Vector costs = new Vector(y.getRowCount());

        List<Vector> yVec = y.asVectors();
        List<Vector> yPrimeVec = yPrime.asVectors();

        costs = costs.apply(yVec, yPrimeVec, vp -> errorCost(vp));

        return Arrays.stream(costs.asArray())
                .average()
                .getAsDouble();
    }

    public abstract double errorCost(Pair<Vector, Vector> pVector);

    public Matrix errorCostPrime(Matrix y, Matrix yPrime) {
        assert y.sameShape(yPrime);

        return new Matrix(y.getRowCount(), y.getColumnCount())
                .apply(y, yPrime, vp -> errorCostPrime(vp));
    }

    public abstract double errorCostPrime(Pair<Double, Double> outputPair);


    public abstract double[][] errorCost(double[][] labels, double[][] output);

    public abstract double[][] errorOutputPrime(double[][] labels, double[][] output, ActivationFunction activationFunction);

}
