package com.mina.mains;

public class Main7 {

    // https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
    public static void main(String[] args) {

        // Handle the Weight Matrix
        // W[Output_Filters][#OfChannels][Kernal_hight][Kernal_weight]

        int filters = 64;
        int channels = 3;
        int kernalHight = 3;
        int kernalWidth = 3;

        int[] bias = new int[filters];


        System.out.println("Fill 4D matrix");
        int[][][][] matrix = new int[filters][channels][kernalHight][kernalWidth];
        int val = 1;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; k++) {
                    for (int l = 0; l < matrix[i][j][k].length; ++l) {
                        matrix[i][j][k][l] = val++;
                    }
                }
            }
        }

        System.out.println("Print out the 4D matrix");
        // print out the matrix
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; k++) {
                    for (int l = 0; l < matrix[i][j][k].length; ++l) {
                        System.out.print(matrix[i][j][k][l] + " ");
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println();
        }

        System.out.println("4D matrix -> 2D matrix");
        // to 2D array
        int[][] out = new int[filters][kernalHight * kernalWidth * channels];
        int col = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; k++) {
                    for (int l = 0; l < matrix[i][j][k].length; ++l) {
                        out[i][col++] = matrix[i][j][k][l];
                    }
                }
            }
            col = 0;
        }

        System.out.println("Print out the 2D matrix");
        // print out the out matrix
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[i].length; j++) {
                System.out.print(out[i][j] + " ");
            }
            System.out.println();
        }

        int n = kernalHight * kernalWidth * filters * channels + bias.length;
        System.out.println("Num of params = " + n);

    }


}
