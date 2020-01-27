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
    private int imageH = 28;
    private int imageW = 28;

    private int filters = 64;
    private Pair<Integer, Integer> kernelSize = new Pair<>(3, 3);


    @Test
    public void matrixPatches_test() {

        double[][][][] temp = new double[size][channels][imageH][imageW];
        generateRandom(temp);
        D4Matrix inputTensor = new D4Matrix(temp);

        int height = kernelSize.getValue0();
        int width = kernelSize.getValue1();
        D4WeightMatrix weight = new D4WeightMatrix(filters, channels, height, width);


        D4Matrix input = inputTensor;
        System.out.println("forwardPropagation:: input shape" + input.shape());

        D3Matrix patches = input.matrixPatches(kernelSize);
        System.out.println("forwardPropagation:: patches shape" + patches.shape());

        System.out.println("forwardPropagation:: weight shape" + weight.shape());

        Matrix K = weight.reshape2D();
        System.out.println("forwardPropagation:: K shape" + K.shape());

        for (int image = 0; image < patches.getSize(); image++) {
            Matrix P = patches.get(image);
            P = P.transpose();
            System.out.println("forwardPropagation::--> P shape" + P.shape());
            Matrix KP = K.dot(P);
            System.out.println("forwardPropagation:: KP shape" + KP.shape());
            System.exit(0);
        }
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