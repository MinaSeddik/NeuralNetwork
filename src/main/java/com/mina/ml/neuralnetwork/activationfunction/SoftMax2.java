package com.mina.ml.neuralnetwork.activationfunction;

import com.mina.ml.neuralnetwork.util.MatrixManipulator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by menai on 2019-01-31.
 */
public class SoftMax2 extends ActivationFunction {

    private final static Logger logger = LoggerFactory.getLogger(SoftMax2.class);

    @Override
    public float[][] activate(float[][] matrix) {
        float[][] result = new float[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            activate(matrix, result, i);
        }

//        MatrixManipulator.printMatrix("SM", result);
        return result;
    }

    private void activate(float[][] matrix, float[][] result, int row) {
        float sum = 0f;
        BigDecimal bd = new BigDecimal(0d);

        System.out.println("*****************************");
        System.out.println("*****************************");
        System.out.println("***************************** row = " + row);



        boolean nan = MatrixManipulator.containNan(result);
        if (nan) {
            System.out.println("SM::activate activate contain Nan");
            System.exit(0);
        }

        for (int col = 0; col < matrix[0].length; col++) {
            System.out.println(" -> " + matrix[row][col] + ", sum before = " + sum + ", will add " + Math.exp(matrix[row][col]));
            sum += Math.exp(matrix[row][col]);
            System.out.println(" ->>>>>>>>>>>>>>>>>>>> " + sum);
            bd = bd.add(new BigDecimal(Math.exp(matrix[row][col])));
            System.out.println(" ->>>>>>>>>>>>>>>>>>>> " + bd);
            System.out.println(" ->>>>>>>>>>>>>>>>>>>> " + new BigDecimal(Math.exp(matrix[row][col])));

//            if (Float.isNaN(sum)){
//                System.out.println("----------------------------------");
//                System.out.println("----------------------------------" + matrix[row][col]);
//                System.out.println("----------------------------------" + Math.exp(matrix[row][col]));
//                System.exit(0);
//            }
        }

        nan = MatrixManipulator.containNan(result);
        if (nan) {
            System.out.println("SM::activate before activate contain Nan");
            System.exit(0);
        }
        System.out.println("SM::activate sum = " + sum);
        System.out.println("SM::activate Bigdecimal sum = " + bd);

        for (int col = 0; col < matrix[0].length; col++) {

            double c = activate(matrix[row][col]);
            String h = Double.toString(c);
//            System.out.println("H = " + h);
             BigDecimal y = new BigDecimal(h);
            System.out.println(y + "/" + bd);
            BigDecimal w = y.divide(bd, 12, RoundingMode.HALF_UP);

            double x = w.doubleValue();
            System.out.println("WWWWWWWWWWwwwwww = " + w);
            System.out.println("XXXXXXXXXXXXXXXX = " + x);

            float f = (float) activate(matrix[row][col]) / sum;
            result[row][col] = f;

            result[row][col] = (float) x;
        }


         nan = MatrixManipulator.containNan(result);
        if (nan) {
            System.out.println("SM::activate after activate contain Nan");
            System.exit(0);
        }
    }

    @Override
    public double activate(float value) {

//        System.out.println("***** value = " + value);
//        System.out.println("***** Math.exp(value) = " + Math.exp(value));
//        if (nan) {
//            System.out.println("SM::activate after activate contain Nan");
//            System.exit(0);
//        }

        return Math.exp(value);
    }

    @Override
    public float activatePrime(float value) {
        // un-defined function
        return 0f;
    }

}
