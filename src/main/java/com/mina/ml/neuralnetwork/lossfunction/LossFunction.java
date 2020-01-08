package com.mina.ml.neuralnetwork.lossfunction;

import com.mina.ml.neuralnetwork.activationfunction.ActivationFunction;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by menai on 2019-01-31.
 */
public abstract class LossFunction {

    private final static Logger logger = LoggerFactory.getLogger(LossFunction.class);

    public double reducedMeanError(float[][] labels, float[][] output) {
        float[][] costs = errorCost(labels, output);

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

    protected abstract float[][] errorCost(float[][] labels, float[][] output);

    public abstract float[][] errorOutputPrime(float[][] labels, float[][] output, ActivationFunction activationFunction);

}
