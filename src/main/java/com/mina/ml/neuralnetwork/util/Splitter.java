package com.mina.ml.neuralnetwork.util;


import org.javatuples.Quartet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Splitter<T> {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Splitter.class);

    private final List<T> x;
    private final List<T> y;

    private final boolean shuffle;

    public Splitter(List<T> x, List<T> y) {
        this(x, y, false);
    }

    public Splitter(List<T> x, List<T> y, boolean shuffle) {
        assert x.size() == y.size();

        this.x = x;
        this.y = y;
        this.shuffle = shuffle;
    }

    public Quartet<List<T>, List<T>, List<T>, List<T>> split(float ratio){
        int splitStartIndex = (int) (x.size() * (1f - ratio));

        List<T> x1 = x.subList(0, splitStartIndex);
        List<T> y1 = y.subList(0, splitStartIndex);

        List<T> x2 = x.subList(splitStartIndex, x.size());
        List<T> y2 = y.subList(splitStartIndex, y.size());

        return new Quartet<>(x1, y1, x2, y2);
    }

    public void reset() {
        if(shuffle){
            shuffle();
        }
    }

    private void shuffle() {
        long seed = System.nanoTime();
        Collections.shuffle(x, new Random(seed));
        Collections.shuffle(y, new Random(seed));
    }
}
