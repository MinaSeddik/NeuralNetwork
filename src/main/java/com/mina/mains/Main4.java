package com.mina.mains;

import com.mina.examples.mnist.MNistReader;
import com.mina.examples.mnist.MNistTraining;
import com.mina.ml.neuralnetwork.factory.Optimizer;
import com.mina.ml.neuralnetwork.layer.Dense;
import com.mina.ml.neuralnetwork.layer.Model;
import com.mina.ml.neuralnetwork.layer.Sequential;
import com.mina.ml.neuralnetwork.layer.Verbosity;
import org.javatuples.Tuple;
import org.javatuples.Unit;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main4 {

    private static final String TRAINING_SET_IMAGES = "train-images-idx3-ubyte";
    private static final String TRAINING_SET_LABELS = "train-labels-idx1-ubyte";
    private static final String TEST_SET_IMAGES = "t10k-images-idx3-ubyte";
    private static final String TEST_SET_LABELS = "t10k-labels-idx1-ubyte";

    private static final String MNIST_DATA_DIR = "mnist/";

    private static final int MNIST_IMAGE_HEIGHT = 28;
    private static final int MNIST_IMAGE_WIDTH = 28;

    private static final int NUM_OF_CLASSES = 10;

    public static void main(String[] args) {

        String imagesTrainingFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TRAINING_SET_IMAGES)
                .getFile();
        String labelsTrainingFile = MNistTraining.class.getClassLoader()
                .getResource(MNIST_DATA_DIR + TRAINING_SET_LABELS)
                .getFile();

        List<int[][]> _images = MNistReader.getImages(imagesTrainingFile);
        int[] _labels = MNistReader.getLabels(labelsTrainingFile);

        List<double[]> images = _images.stream().map(matrix -> convert2DataArray(matrix)).collect(Collectors.toList());
        List<double[]> labels = Arrays.stream(_labels).boxed().map(l -> convert2HotEncodedArray(l)).collect(Collectors.toList());



        Tuple inputShape = new Unit(28 * 28);
        Model model = new Sequential(new Dense[]{
                new Dense(64, inputShape, "relu"),
                new Dense(64, "relu"),
                new Dense(10, "softmax")
        });

        model.summary(line -> System.out.println(line));

        double learningRate = 0.001;
        Optimizer optimizer = new Optimizer(learningRate);
        model.compile(optimizer, "categorical_crossentropy", "");

        model.fit(images, labels, 0.1f, true, 128, 1000, Verbosity.VERBOSE, null);


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
}
