package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MatrixLogger {

    private final static Logger logger = LoggerFactory.getLogger(MatrixLogger.class);

    public static String simulate(String label, int rows, int cols, int startColIndex, char r, char c) {

        StringBuffer matrixAsString = new StringBuffer(label + "\n");

        boolean rowPrinted;
        boolean colPrinted;
        int voidRowsCount = 0;
        String val;
        for (int x = 0; x < rows; x++) {
            colPrinted = false;
            rowPrinted = false;
            for (int y = 0; y < cols+startColIndex; y++) {
                if ((x == 0 || x == 1 || x == 2 || x + 1 == rows || x + 2 == rows) &&
                        (y <= 3  || y + startColIndex == cols || y + startColIndex + 1 == cols) ) {

                    val = c == Character.MIN_VALUE ? String.format("%c%d-%d\t", r, x, y):
                            String.format("%c%d%c%d\t", r, x, c, y+startColIndex);
                    rowPrinted = true;
                } else if (rowPrinted && !colPrinted) {
                    val = "...  ";
                    colPrinted = true;
                } else if (!rowPrinted && !colPrinted && voidRowsCount < 3) {
                    val = "..\t..";
                    rowPrinted = true;
                    colPrinted = true;
                    voidRowsCount++;
                } else {
                    val = "";
                }
                matrixAsString.append(val);
            }
            matrixAsString.append(rowPrinted || colPrinted || voidRowsCount < 3 ? "\n" : "");
        }
        return matrixAsString.toString();
    }

    public static String simulate(String label, int rows, int cols, char r, char c) {
        return simulate(label, rows, cols, 0, r, c);
    }

}
