package com.mina.ml.neuralnetwork.util;

@FunctionalInterface
public interface TensorConsumer {
    void accept(int startIndex, int endIndex);
}
