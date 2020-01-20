package com.mina.preprocessing;

import org.apache.commons.collections4.MapUtils;

import java.util.Arrays;
import java.util.Map;

public class OneHotEncoder {


    private int numOfClasses = 0;

    public OneHotEncoder() {
    }

    public OneHotEncoder(int numOfClasses) {
        this.numOfClasses = numOfClasses;
    }

    public double[][] fitTransform(int[] values) {
        if (numOfClasses == 0) {
            numOfClasses = Arrays.stream(values).max().getAsInt();
        }

        return transform(values);
    }

    public double[][] transform(int[] values) {
        double[][] labels = new double[values.length][numOfClasses];
        for (int i = 0; i < values.length; i++) {
            labels[i][values[i]] = 1d;
        }

        return labels;
    }

    public double[] transform(int value) {
        double[] label = new double[numOfClasses];
        label[value] = 1d;

        return label;
    }

    public int inverseTransform(double[] value) {
        for (int i=0;i<value.length;i++){
            if( value[i] == 1d ){
                return i;
            }
        }

        return -1;
    }

    public int[] inverseTransform(double[][] values) {
        int[] output = new int[values.length];

        for(int i=0;i<values.length;i++){
            output[i] = inverseTransform(values[i]);
        }

        return output;
    }
}
