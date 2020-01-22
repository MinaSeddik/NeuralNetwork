package com.mina.ml.neuralnetwork.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class FilesUtil {

    private static final long serialVersionUID = 6529685098267757690L;
    private final static Logger logger = LoggerFactory.getLogger(FilesUtil.class);

    public static void serializeData(String fileName, Object data) {
        try (FileOutputStream fos = new FileOutputStream(new File(fileName))) {
            ObjectOutputStream objectOut = new ObjectOutputStream(fos);
            objectOut.writeObject(data);
            objectOut.close();

        } catch (IOException ex) {
            logger.error("{}, Exception: {}", ex.getMessage(), ex);
            throw new RuntimeException(ex.getMessage());
        }
    }

    public static <T> T deSerializeData(String fileName) {
        T model;
        try (FileInputStream fis = new FileInputStream(new File(fileName))) {
            ObjectInputStream objectIn = new ObjectInputStream(fis);
            model = (T) objectIn.readObject();
            objectIn.close();
        } catch (IOException | ClassNotFoundException ex) {
            logger.error("{}, Exception: {}", ex.getMessage(), ex);
            throw new RuntimeException(ex.getMessage());
        }

        return model;
    }
}
