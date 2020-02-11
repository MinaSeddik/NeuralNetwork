package com.mina.ml.neuralnetwork;

/**
 * Created by menai on 2019-02-01.
 */
public interface Constants {

    String NUMBER_OF_FEATURES = "feature.nodes.count";

    String NUMBER_OF_OUTPUT_NODES = "output.nodes.count";
    String OUTPUT_ACTIVATION_FUNCTION = "output.nodes.activation";

    String NUMBER_OF_HIDDEN_LAYERS = "hidden.count";
    String NUMBER_OF_HIDDEN_LAYER_NODES = "hidden.nodes.count.{HIDDEN_LAYER}";
    String HIDDEN_ACTIVATION_FUNCTION = "hidden.nodes.activation.{HIDDEN_LAYER}";

    String LEARNING_RATE = "learning.rate";
    String LEARNING_RATE_MECHANISM = "learning.rate.mechanism";

    String LOSS_FUNCTION = "loss.function";

    String BATCH_SIZE = "batch.size";
    String MAX_EPOCH = "epoch.max";

    String VOID_ACTIVATION_FUNCTION = "void";
    String RELU_ACTIVATION_FUNCTION = "relu";
    String SIGMOID_ACTIVATION_FUNCTION = "sigmoid";
    String TANSH_ACTIVATION_FUNCTION = "tanh";
    String SOFT_MAX_ACTIVATION_FUNCTION = "softmax";

    String MEAN_SQUARED_ERROR_LOSS_FUNCTION = "MeanSquaredError";
    String CROSS_ENTROPY_LOSS_FUNCTION = "CrossEntropyLoss";
    String BINARY_CROSS_ENTROPY_LOSS_FUNCTION = "binary_crossentropy";
    String CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION = "categorical_crossentropy";

}
