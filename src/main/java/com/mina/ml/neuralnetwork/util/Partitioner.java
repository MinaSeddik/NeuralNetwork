package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class Partitioner {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Partitioner.class);

    private List<? extends Object> x;
    private List<? extends Object> y;
    private int batchSize;

    private int startBatchIndex = 0;

    public Partitioner(List<? extends Object> x, List<? extends Object> y, int batchSize) {
        assert x.size() == y.size();

        this.x = x;
        this.y = y;
        this.batchSize = batchSize;
    }

    public boolean hasNext() {
        return startBatchIndex < x.size();
    }

    public Pair<List<? extends Object>, List<? extends Object>> getNext() {
        int endIndex = Math.min(startBatchIndex + batchSize, x.size()) ;
        List<? extends Object> xList = x.subList(startBatchIndex, endIndex);
        List<? extends Object> yList = y.subList(startBatchIndex, endIndex);
        startBatchIndex = endIndex;

        return new Pair<>(xList, yList);
    }

}
