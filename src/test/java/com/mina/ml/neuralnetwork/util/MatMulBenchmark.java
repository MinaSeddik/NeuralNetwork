package com.mina.ml.neuralnetwork.util;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;


@State(Scope.Benchmark)
public class MatMulBenchmark {

    private final static int ROWS1 = 128;
    private final static int COLS1 = 10;

    private final static int ROWS2 = 10;
    private final static int COLS2 = 65;

    private double[][] matrix1 = new double[ROWS1][COLS1];
    private double[][] matrix2 = new double[ROWS2][COLS2];

    @Setup(Level.Trial)
    public void doSetup() {

        generateRandom(matrix1);
        generateRandom(matrix2);

    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }


    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Fork(value = 3, warmups = 2)
    @Warmup(iterations = 5, timeUnit = TimeUnit.NANOSECONDS)
    @Measurement(iterations = 4, timeUnit = TimeUnit.NANOSECONDS)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void matrixMultiplication_SingleThread(Blackhole bh) {
        double[][] matrix = MatrixManipulator.multiply_singleThread(matrix1, matrix2);
        bh.consume(matrix);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Fork(value = 3, warmups = 2)
    @Warmup(iterations = 5, timeUnit = TimeUnit.NANOSECONDS)
    @Measurement(iterations = 4, timeUnit = TimeUnit.NANOSECONDS)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void matrixMultiplication_MultiThreading(Blackhole bh) {
        double[][] matrix = MatrixManipulator.multiply(matrix1, matrix2);
        bh.consume(matrix);
    }

    private void generateRandom(double[][] matrix) {
        Random r = new Random();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = r.nextDouble();
            }
        }
    }

}