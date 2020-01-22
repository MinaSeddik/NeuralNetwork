package com.mina.ml.neuralnetwork.util;

@FunctionalInterface
public interface CollectionConsumer {
    void accept(int startIndex, int endIndex);
}
