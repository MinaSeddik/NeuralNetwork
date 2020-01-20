package com.mina.preprocessing;

public interface Encoder<T> {

    double[][] fitTransform(T[] values);
    T inverseTransform(double[][] values);
}
