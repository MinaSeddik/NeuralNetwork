package com.mina.ml.neuralnetwork.util;

import org.javatuples.Pair;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by menai on 2020-01-26.
 */
public class D4MatrixTest {

    private int size = 128;
    private int channels = 3;
    private int imageH = 224;
    private int imageW = 224;


    @Test
    public void matrixPatches_test() {

        double[][][][] mat = new double[size][channels][imageH][imageW];
        generateRandom(mat);

        Pair<Integer, Integer> window = new Pair<>(3, 3);

        D4Matrix matrix = new D4Matrix(mat);

        D3Matrix result = matrix.matrixPatches(window);
//        D3Matrix result = matrix.matrixPatches_test(window);

        System.out.println(result.shape());

    }

    private void generateRandom(double[][][][] matrix) {
        Random r = new Random();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < channels; j++) {
                for (int k = 0; k < imageH; k++) {
                    for (int l = 0; l < imageW; l++) {
                        matrix[i][j][k][l] = r.nextDouble();
                    }
                }
            }
        }
    }

}