package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.util.FilesUtil;
import com.mina.ml.neuralnetwork.util.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class ModelCheckpoint {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(ModelCheckpoint.class);

    public static final String VALIDATION_ACCURACY = "val_accuracy";  // default monitor
    public static final String VALIDATION_LOSS = "val_loss";
    public static final String TRAIN_ACCURACY = "train_accuracy";
    public static final String TRAIN_LOSS = "train_loss";

    private static final String MAX_MODE = "max";   // default mode
    private static final String MIN_MODE = "min";

    private Double lastEpochValue = null;

    private final String fileNamePattern;
    private final String monitor;
    private final Verbosity verbosity;
    private final boolean saveBestOnly;
    private final String mode;

    public ModelCheckpoint(String fileNamePattern) {
        this(fileNamePattern, VALIDATION_ACCURACY, Verbosity.NORMAL, true, MAX_MODE);
    }

    public ModelCheckpoint(String fileNamePattern, String monitor, Verbosity verbosity, boolean saveBestOnly, String mode) {
        this.fileNamePattern = fileNamePattern;
        this.monitor = monitor;
        this.verbosity = verbosity;
        this.saveBestOnly = saveBestOnly;
        this.mode = mode;
    }

    public void handle(Map<String, Object> params, List<? extends Layer> layers) {
        double currentValue = (double) params.get(monitor);
        if (lastEpochValue == null) {
            lastEpochValue = currentValue;
            return;
        }

        // save the model
        if ((mode.equals(MIN_MODE) && currentValue < lastEpochValue) ||
                (mode.equals(MAX_MODE) && currentValue > lastEpochValue)) {
            // save the weights to the file specified
            String file = getFileName(params);
            Map<Integer, Tensor> modelWeights = layers.stream()
                    .collect(Collectors.toMap(Layer::getIndex, Layer::getWeights));
            FilesUtil.serializeData(file, modelWeights);

            System.out.println(String.format("Epoch %d: %s improved from %.4f to %.4f, saving model to %s",
                    params.get("epoch"), monitor, lastEpochValue, currentValue, new File(file).getName()));
            lastEpochValue = currentValue;
        } else {
            System.out.println(String.format("Epoch %02d: %s did not improve.", params.get("epoch"), monitor));
        }

    }

    private String getFileName(Map<String, Object> params) {
        String fileName = fileNamePattern;

        Pattern pattern = Pattern.compile("\\{([^{}]+)\\}");
        Matcher matcher = pattern.matcher(fileNamePattern);
        while (matcher.find()) {
            String placeholder = matcher.group();
            String temp = placeholder.replaceAll("\\{", "")
                    .replaceAll("\\}", "");

            String placeholderValue;
            if (temp.contains(":")) {
                String placeholderName = temp.split(":")[0];
                String placeholderFormat = "%" + temp.split(":")[1];
                placeholderValue = String.format(placeholderFormat, params.get(placeholderName));
            } else {
                placeholderValue = params.get(temp).toString();
            }
            fileName = fileName.replace(placeholder, placeholderValue);
        }

        return fileName;
    }

}
