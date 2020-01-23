package com.mina.ml.neuralnetwork.util;

public abstract class Tensor extends TensorParallelizer {

    private static final long serialVersionUID = 6529685098267757690L;
//    private final static Logger logger = LoggerFactory.getLogger(Tensor.class);

    public abstract int getSize();
}
