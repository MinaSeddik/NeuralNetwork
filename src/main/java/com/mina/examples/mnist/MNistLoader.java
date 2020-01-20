package com.mina.examples.mnist;

import com.mina.preprocessing.OneHotEncoder;
import org.javatuples.Quartet;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

public class MNistLoader {

    private static final String BASE_URL = "http://yann.lecun.com/exdb/mnist/";

    private static final String TRAINING_SET_IMAGES = "train-images-idx3-ubyte";
    private static final String TRAINING_SET_LABELS = "train-labels-idx1-ubyte";
    private static final String TEST_SET_IMAGES = "t10k-images-idx3-ubyte";
    private static final String TEST_SET_LABELS = "t10k-labels-idx1-ubyte";

    private static final String MNIST_DATA_DIR = "mnist/";

    private static final int MNIST_IMAGE_HEIGHT = 28;
    private static final int MNIST_IMAGE_WIDTH = 28;

    private static final int NUM_OF_CLASSES = 10;

    public Quartet<List<double[]>, List<double[]>, List<double[]>, List<double[]>> loadMNistDataSet() {

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

        // training data set
        String imagesTrainingFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TRAINING_SET_IMAGES)
                .getFile();
        String labelsTrainingFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TRAINING_SET_LABELS)
                .getFile();

        // test data set
        String imagesTestFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TEST_SET_IMAGES)
                .getFile();
        String labelsTestFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TEST_SET_LABELS)
                .getFile();

        List<int[][]> _trainingImages = MNistReader.getImages(imagesTrainingFile);
        int[] _trainingLabels = MNistReader.getLabels(labelsTrainingFile);
        List<int[][]> _testImages = MNistReader.getImages(imagesTestFile);
        int[] _testLabels = MNistReader.getLabels(labelsTestFile);

        List<double[]> x_train = _trainingImages.stream().map(matrix -> convert2DataArray(matrix)).collect(Collectors.toList());
        List<double[]> x_test = _testImages.stream().map(matrix -> convert2DataArray(matrix)).collect(Collectors.toList());

        OneHotEncoder encoder = new OneHotEncoder(NUM_OF_CLASSES);
        List<double[]> y_train = Arrays.stream(_trainingLabels).boxed().map(l -> encoder.transform(l)).collect(Collectors.toList());
        List<double[]> y_test = Arrays.stream(_testLabels).boxed().map(l -> encoder.transform(l)).collect(Collectors.toList());

        return new Quartet<>(x_train, y_train, x_test, y_test);
    }

    private boolean mNistDataSetExists() {
        return MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TRAINING_SET_IMAGES) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TRAINING_SET_LABELS) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TEST_SET_IMAGES) != null &&
                MNistTraining.class.getClassLoader().getResource(MNIST_DATA_DIR + TEST_SET_LABELS) != null;
    }

    private void downloadMNistDataSetGunZip(String gzFileRemoteUrl, String fileName) throws Exception {
        ClassLoader classLoader = MNistTraining.class.getClassLoader();

        String gzLocalFilePath = MNIST_DATA_DIR + fileName + ".gz";
        InputStream in = new URL(gzFileRemoteUrl).openStream();
        Files.copy(in, Paths.get(gzLocalFilePath), StandardCopyOption.REPLACE_EXISTING);

        URL url = classLoader.getResource(gzLocalFilePath);
        gunzip(new File(url.getFile()));
    }

    private void gunzip(File file) {
        File outputFile = new File(file.getParent(), file.getName().replace(".gz", ""));
        try (GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(file))) {
            Files.copy(gzis, Paths.get(outputFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private static double[] convert2DataArray(int[][] matrix) {
        double[] dataImage = new double[MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH];

        int d = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                dataImage[d++] = matrix[i][j];// / 255.0d;
            }
        }
        return dataImage;
    }
}
