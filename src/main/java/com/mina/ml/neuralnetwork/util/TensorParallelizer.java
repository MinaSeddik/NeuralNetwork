package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class TensorParallelizer implements Serializable {

    private static final long serialVersionUID = 6529685098267757690L;
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

    public void parallelizeOperation(TensorConsumer tensorConsumer) {
        if (getSize() < NUM_OF_THREADS) {
            tensorConsumer.accept(0, getSize());
            return;
        }

        int batchSize = getSize() / NUM_OF_THREADS;
//        System.out.println("getSize() = " +getSize());
//        System.out.println("NUM_OF_THREADS = " + NUM_OF_THREADS);
//        System.out.println("batchSize = " + batchSize);

        List<Future<?>> futures = IntStream.range(0, NUM_OF_THREADS)
                .mapToObj(i -> {

                    int startIndex = i * batchSize;
                    int endIndex = i == (NUM_OF_THREADS-1) ? getSize() : i * batchSize + batchSize;

                    Future<?> future = executor.submit(() -> tensorConsumer.accept(startIndex, endIndex));

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
