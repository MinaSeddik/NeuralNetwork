package com.mina.ml.neuralnetwork.factory;

import com.mina.ml.neuralnetwork.lossfunction.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collection;

import static com.mina.ml.neuralnetwork.Constants.*;

/**
 * Created by menai on 2019-02-01.
 */
public class LossFunctionFactory {

    private final static Logger logger = LoggerFactory.getLogger(LossFunctionFactory.class);

    private Collection<String> lossFunctions = Arrays.asList(
            MEAN_SQUARED_ERROR_LOSS_FUNCTION,
            CROSS_ENTROPY_LOSS_FUNCTION,
            BINARY_CROSS_ENTROPY_LOSS_FUNCTION,
            CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION
    );

    public LossFunction createLossFunction(String lossFunctionName) throws Exception {

        if (!lossFunctions.contains(lossFunctionName)) {
            logger.error("Invalid Loss Function Name '" + lossFunctionName + "'");
            throw new Exception("Invalid Loss Function Name '" + lossFunctionName + "'");
        }

        LossFunction lossFunction = null;
        switch (lossFunctionName) {
            case MEAN_SQUARED_ERROR_LOSS_FUNCTION:
                lossFunction = new MeanSquaredError();
                break;
            case CROSS_ENTROPY_LOSS_FUNCTION:
                lossFunction = new CrossEntropyLoss();
                break;
            case BINARY_CROSS_ENTROPY_LOSS_FUNCTION:
                lossFunction = new BinaryCrossEntropyLoss();
                break;
            case CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION:
                lossFunction = new CategoricalCrossEntropyLoss();
                break;
        }

        return lossFunction;
    }

}
