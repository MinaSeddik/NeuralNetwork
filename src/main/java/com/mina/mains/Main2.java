package com.mina.mains;

import com.mina.ml.neuralnetwork.layer.Dense;
import org.javatuples.Unit;

public class Main2 {


    public static void main(String[] args) {

        Unit<Integer> pair = new Unit<>(24);
        Dense ad = new Dense(5, pair, "thisIsActivation");
    }
    
}
