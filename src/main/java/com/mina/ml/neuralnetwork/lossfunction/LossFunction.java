package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.Matrix;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import com.mina.ml.neuralnetwork.util.Vector;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

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


    public double reducedMeanError2(Matrix y, Matrix yPrime) {
        Vector costs = errorCost(y, yPrime);
        return Arrays.stream(costs.asArray()).average().getAsDouble();
    }

    private Vector errorCost(Matrix y, Matrix yPrime){
        assert y.sameShape(yPrime);

        Vector costs = new Vector(y.getRowCount());

        List<Vector> yVec = y.asVectors();
        List<Vector> yPrimeVec = yPrime.asVectors();

        Function<Pair<Vector, Vector>, Double> function = vp-> errorCost(vp);
        costs.apply(yVec, yPrimeVec, function);

        return costs;
    }

    public abstract double errorCost(Pair<Vector, Vector> pVector);

    protected abstract double[][] errorCost(double[][] labels, double[][] output);

    public abstract double[][] errorOutputPrime(double[][] labels, double[][] output, ActivationFunction activationFunction);

}
