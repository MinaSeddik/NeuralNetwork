package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;

import java.util.List;

public class Partitioner<T> {

    private List<T> x;
    private List<T> y;
    private int batchSize;

    private int startBatchIndex = 0;

    public Partitioner(List<T> x, List<T> y, int batchSize) {
        assert x.size() == y.size();

        this.x = x;
        this.y = y;
        this.batchSize = batchSize;
    }

    public boolean hasNext() {
        return startBatchIndex < x.size();
    }

    public Pair<List<T>, List<T>> getNext() {
        int endIndex = Math.min(startBatchIndex + batchSize, x.size()) ;
        List<T> xList = x.subList(startBatchIndex, endIndex);
        List<T> yList = y.subList(startBatchIndex, endIndex);
        startBatchIndex = endIndex;

        return new Pair<>(xList, yList);
    }

}
