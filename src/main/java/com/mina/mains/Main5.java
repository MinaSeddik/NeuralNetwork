package com.mina.mains;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class Main5 {

    public static void main(String[] args) {

        List<String> x = Arrays.asList("apple1", "lion1", "soft1", "Montreal1", "1_1", "desk1", "Mina1", "MS1", "123", "XXX1", "YYY1", "ZZZ1", "@1", "Peanut1", "Cheese1", "Wine1");
        List<String> y = Arrays.asList("apple2", "lion2", "soft2", "Montreal2", "1_2", "desk2", "Mina2", "MS2", "123", "XXX2", "YYY2", "ZZZ2", "@2", "Peanut2", "Cheese2", "Wine2");

        long seed = System.nanoTime();
        Collections.shuffle(x, new Random(seed));
        Collections.shuffle(y, new Random(seed));

        IntStream.range(0, x.size()).forEach(i -> System.out.println(x.get(i) + "   " + y.get(i)));
    }
}
