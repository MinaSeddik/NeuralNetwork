package com.mina.ml.neuralnetwork.layer;

import com.mina.ml.neuralnetwork.factory.Optimizer;
import org.javatuples.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.List;
import java.util.function.Consumer;

public abstract class Model implements Serializable {

    private final static Logger logger = LoggerFactory.getLogger(Model.class);

    public abstract void summary(Consumer consumer);

    public abstract void compile(Optimizer optimizer, String loss, String metrics);

    public abstract void fit(List<double[]> xTrain, List<double[]> yTrain, float validationSplit, boolean shuffle, int batchSize, int epochs, Verbosity verbosity, List<ModelCheckpoint> callbacks);

    public abstract Pair<Double, Double> evaluate(List<double[]> xTest, List<double[]> yTest);

    public void save(String modelFilePath) {

//        System.out.println("inside save " + modelFilePath);
        try (FileOutputStream fos = new FileOutputStream(new File(modelFilePath))) {
            ObjectOutputStream objectOut = new ObjectOutputStream(fos);
            objectOut.writeObject(this);
            objectOut.close();
//            System.out.println("The Object  was successfully written to a file");

        } catch (IOException ex) {
            logger.error("{}, Exception: {}", ex.getMessage(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    public static Model load(String modelFilePath) {

        Model model;
//        System.out.println("inside load " + modelFilePath);
        try (FileInputStream fis = new FileInputStream(new File(modelFilePath))) {
            ObjectInputStream objectIn = new ObjectInputStream(fis);
            model = (Model) objectIn.readObject();
            objectIn.close();
//            System.out.println("The Object  was successfully read from a file");

        } catch (IOException | ClassNotFoundException ex) {
            logger.error("{}, Exception: {}", ex.getMessage(), ex);
            throw new RuntimeException(ex.getMessage());
        }

        return model;
    }

//    rounded_predictions = model.predict_classes(x_test);
//    predictions = model.predict(x_test);

}
