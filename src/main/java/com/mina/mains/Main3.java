package com.mina.mains;

import java.util.Random;
import java.util.stream.IntStream;

public class Main3 {

    public static void main(String[] args) {

        double minRange = -1.0d;
        double maxRange = 1.0d;
        Random random = new Random();
        random.setSeed(0);

        IntStream.range(0, 10).forEach(i ->
                System.out.println(minRange + (maxRange - minRange) * random.nextDouble()));

    }
}
