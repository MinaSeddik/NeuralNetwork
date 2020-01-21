package com.mina.ml.neuralnetwork.layer;

import org.apache.commons.lang3.ObjectUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class ModelCheckpoint {

    private final static Logger logger = LoggerFactory.getLogger(ModelCheckpoint.class);

    private static final String VALIDATION_ACCURACY = "val_accuracy";  // default monitor
    private static final String VALIDATION_LOSS = "val_loss";
    private static final String TRAIN_ACCURACY = "train_accuracy";
    private static final String TRAIN_LOSS = "train_loss";

    private static final String MAX_MODE = "max";   // default mode
    private static final String MIN_MODE = "min";

    private Double lastEpochValue = null;

    private final String filePath;
    private final String monitor;
    private final Verbosity verbosity;
    private final boolean saveBestOnly;
    private final String mode;

    private Map<String, Consumer<Double>> map = new HashMap<>();

    public ModelCheckpoint(String filePath) {
        this(filePath, VALIDATION_ACCURACY, Verbosity.NORMAL, true, MAX_MODE);
    }

    public ModelCheckpoint(String filePath, String monitor, Verbosity verbosity, boolean saveBestOnly, String mode) {
        this.filePath = filePath;

        this.monitor = monitor;
        map.put("val_accuracy", value -> handle(value));
        map.put("val_loss", value -> handle(value));
        map.put("train_accuracy", value -> handle(value));
        map.put("train_loss", value -> handle(value));


        this.verbosity = verbosity;
        this.saveBestOnly = saveBestOnly;
        this.mode = mode;
    }

    private void handle(Double currentValue) {
        if(lastEpochValue == null){
            lastEpochValue = currentValue;
            return;
        }

        switch (mode){
            case MIN_MODE:
                if(currentValue < lastEpochValue){
                    // save the weights to the file specified

                    lastEpochValue = currentValue;
                }
                break;
            case MAX_MODE:
            default:
               if(currentValue > lastEpochValue){
                   // save the weights to the file specified

                   lastEpochValue = currentValue;
               }
        }

    }


    public void handle(double trainLoss, double trainAcc, double valLoss, double valAcc) {

        switch (monitor) {
            case VALIDATION_LOSS:
                handle(valLoss);
                break;
            case TRAIN_ACCURACY:
                handle(trainAcc);
                break;
            case TRAIN_LOSS:
                handle(trainLoss);
                break;
            case VALIDATION_ACCURACY:
            default:
                handle(valAcc);
        }
    }


}
