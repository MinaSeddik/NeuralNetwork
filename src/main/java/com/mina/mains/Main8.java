package com.mina.mains;

public class Main8 {

    // https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
    public static void main(String[] args) {

        // Handle the Image patches
        int imgWidth = 4;
        int imgHight = 4;
        int channels = 3;

        int kernalWidth = 2;
        int kernalHight = 2;

        int P = ( imgWidth - kernalWidth + 1 ) * ( imgHight - kernalHight + 1 );
        System.out.println("Patch Count = " + P);

        int K = kernalWidth * kernalHight;
        System.out.println("Patch Count = " + K);

        int[][][] image = new int[channels][imgHight][imgWidth];
        System.out.println("Fill 3D Image");
        int val = 1;
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                for (int k = 0; k < image[i][j].length; k++) {
                    image[i][j][k] = val++;
                }
            }
        }

        System.out.println("Print out the 3D Image");
        // print out the matrix
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                for (int k = 0; k < image[i][j].length; k++) {
                    System.out.print(image[i][j][k]+ " ");
                }
                System.out.println();
            }
            System.out.println();
        }


    }

}
