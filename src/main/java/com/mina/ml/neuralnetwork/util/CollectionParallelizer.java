package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class CollectionParallelizer<T> {

    private final static Logger logger = LoggerFactory.getLogger(Matrix.class);

    private static final int NUM_OF_THREADS = Runtime.getRuntime().availableProcessors();

    // Make the executor a daemon thread so that it will get killed automatically when the main program exits.
    private static final ExecutorService executor = Executors.newFixedThreadPool(NUM_OF_THREADS,new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    });

    protected T collection;

    public void parallelizeOperation(CollectionConsumer collectionConsumer) {
        if (getSize() < NUM_OF_THREADS) {
            collectionConsumer.accept(0, getSize());
            return;
        }

        int batchSize = getSize() / NUM_OF_THREADS;
        List<Future<?>> futures = IntStream.range(0, NUM_OF_THREADS)
                .mapToObj(i -> {

                    int startIndex = i * batchSize;
                    int endIndex = i == NUM_OF_THREADS ? getSize() : i * batchSize + batchSize;

                    Future<?> future = executor.submit(() -> collectionConsumer.accept(startIndex, endIndex));

                    return future;
                }).collect(Collectors.toList());

        futures.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                logger.error("Exception: " + ex);
                throw new RuntimeException("Exception: " + ex.getClass() + " " + ex.getMessage());
            }
        });
    }

    public abstract int getSize();
}
