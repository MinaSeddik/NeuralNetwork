package com.mina.mains;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class Main6 {


    public static void main(String[] args) {

        int[][] matrix = new int[][]{{1,2,3}, {4,5,6}, {7,8,9}};

        Arrays.stream(matrix).forEach(x -> System.out.println(x));

    }

}
