package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public abstract class Tensor extends TensorParallelizer {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(Tensor.class);

    public abstract int getSize();

    public abstract String shape();

    public abstract boolean sameShape(Tensor tensor);

    public static Tensor getTensor(List<? extends Object> data) {

        if (data.get(0) instanceof double[]) {
            return new Matrix(((List<double[]>) data));
        } else if (data.get(0) instanceof double[][]) {
            return new D3Matrix(((List<double[][]>) data));
        } else if (data.get(0) instanceof double[][][]) {
            return new D4Matrix(((List<double[][][]>) data));
        } else {
            String message = "UnSupported Data type " + data.get(0).getClass();
            logger.error(message);
            throw new RuntimeException(message);
        }

    }
}
