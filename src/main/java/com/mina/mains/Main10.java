package com.mina.mains;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main10 {

    public static void main(String[] args) {
        AtomicInteger i = new AtomicInteger();

        List<String> names = Stream.generate(() -> "String_" + i.getAndIncrement())
                .limit(35000)
                .collect(Collectors.toList());

//        Spliterator<String> split1 = Executor.generateElements().spliterator();
//        Spliterator<String> split2 = split1.trySplit();


        // Getting Spliterator
//        Spliterator<String> namesSpliterator = names.spliterator();


    }


}
