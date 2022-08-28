package com.mina.examples.mnist;

import com.mina.ml.neuralnetwork.Configuration;
import com.mina.ml.neuralnetwork.Constants;
import com.mina.ml.neuralnetwork.NeuralNetwork;
import com.mina.ml.neuralnetwork.util.MatrixManipulator;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

/**
 * Created by menai on 2019-02-13.
 */
public class MNistTraining {

    private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";

    private static final String TRAINING_SET_IMAGES = "train-images-idx3-ubyte";
    private static final String TRAINING_SET_LABELS = "train-labels-idx1-ubyte";
    private static final String TEST_SET_IMAGES = "t10k-images-idx3-ubyte";
    private static final String TEST_SET_LABELS = "t10k-labels-idx1-ubyte";

    private static final String MNIST_DATA_DIR = "/mnist/";

    private static final int MNIST_IMAGE_HEIGHT = 28;
    private static final int MNIST_IMAGE_WIDTH = 28;

    private static final int NUM_OF_CLASSES = 10;


    public static void main(String[] args) throws IOException {

        if (!mNistDataSetExists()) {
            try {
                downloadMNistDataSetGunZip(BASE_URL + TRAINING_SET_IMAGES + ".gz", TRAINING_SET_IMAGES);
                downloadMNistDataSetGunZip(BASE_URL + TRAINING_SET_LABELS + ".gz", TRAINING_SET_LABELS);
                downloadMNistDataSetGunZip(BASE_URL + TEST_SET_IMAGES + ".gz", TEST_SET_IMAGES);
                downloadMNistDataSetGunZip(BASE_URL + TEST_SET_LABELS + ".gz", TEST_SET_LABELS);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

//        String imagesTrainingFile = MNistTraining.class.getClassLoader()
//                .getResource(MNIST_DATA_DIR + TRAINING_SET_IMAGES)
//                .getFile();
//        String labelsTrainingFile = MNistTraining.class.getClassLoader()
//                .getResource(MNIST_DATA_DIR + TRAINING_SET_LABELS)
//                .getFile();

//        List<int[][]> _images = MNistReader.getImages(imagesTrainingFile);
//        int[] _labels = MNistReader.getLabels(labelsTrainingFile);

        InputStream imagesTrainingInputStream = MNistTraining.class
                .getResourceAsStream(MNIST_DATA_DIR + TRAINING_SET_IMAGES);
        InputStream labelsTrainingInputStream = MNistTraining.class
                .getResourceAsStream(MNIST_DATA_DIR + TRAINING_SET_LABELS);

        assert imagesTrainingInputStream != null;
        assert labelsTrainingInputStream != null;

        List<int[][]> _images = MNistReader.getImagesFromStream(imagesTrainingInputStream);
        int[] _labels = MNistReader.getLabelsFromStream(labelsTrainingInputStream);

        List<double[]> images = _images.stream().map(matrix -> convert2DataArray(matrix)).collect(Collectors.toList());
        List<double[]> labels = Arrays.stream(_labels).boxed().map(l -> convert2HotEncodedArray(l)).collect(Collectors.toList());

        Properties neuralNetworkProperties = new Properties();

        neuralNetworkProperties.put(Constants.NUMBER_OF_FEATURES, 28*28);

        neuralNetworkProperties.put(Constants.NUMBER_OF_OUTPUT_NODES, 10);
        neuralNetworkProperties.put(Constants.OUTPUT_ACTIVATION_FUNCTION, "softmax");
//        neuralNetworkProperties.put(Constants.OUTPUT_ACTIVATION_FUNCTION, "relu");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYERS, 2);

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "1"), 64);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "1"), "relu");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "2"), 64);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "2"), "relu");


        neuralNetworkProperties.put(Constants.LEARNING_RATE, 0.001);

        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "CrossEntropyLoss");
//        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "MeanSquaredError");

        neuralNetworkProperties.put(Constants.BATCH_SIZE, 128);
        neuralNetworkProperties.put(Constants.MAX_EPOCH, 100);

        Configuration configuration = new Configuration(neuralNetworkProperties);
        try {
            NeuralNetwork neuralNetwork = new NeuralNetwork(configuration);

            neuralNetwork.fetchDataSet(images, labels);
            neuralNetwork.train();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static double[] convert2DataArray(int[][] matrix) {
        double[] dataImage = new double[MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH];

        int d = 0;
        for(int i=0;i<matrix.length;i++){
            for(int j=0;j<matrix[0].length;j++){
                dataImage[d++] = matrix[i][j] / 255.0d;
            }
        }
        return dataImage;
    }

    private static double[] convert2HotEncodedArray(int label) {
        double[] hotEncodedLabel = new double[NUM_OF_CLASSES];
        hotEncodedLabel[label] = 1d;
        return hotEncodedLabel;
    }

    private static boolean mNistDataSetExists() {
        return MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TRAINING_SET_IMAGES) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TRAINING_SET_LABELS) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TEST_SET_IMAGES) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TEST_SET_LABELS) != null;
    }

    private static void downloadMNistDataSetGunZip(String gzFileRemoteUrl, String fileName) throws Exception {
        InputStream in = new URL(gzFileRemoteUrl).openStream();

        File mnistTempDir = new File(System.getProperty("java.io.tmpdir")   , MNIST_DATA_DIR);
        if (! mnistTempDir.exists()){
            mnistTempDir.mkdir();
        }

        File mnistTempFile = new File(mnistTempDir, fileName + ".gz");
        Files.copy(in, Paths.get(mnistTempFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);

        gunzip(mnistTempFile);
    }

    private static void gunzip(File file) {
        File outputFile = new File(file.getParent(), file.getName().replace(".gz", ""));

        try (GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(file))) {
            Files.copy(gzis, Paths.get(outputFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}
