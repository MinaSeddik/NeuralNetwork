import com.mina.ml.neuralnetwork.Configuration;
import com.mina.ml.neuralnetwork.Constants;
import com.mina.ml.neuralnetwork.NeuralNetwork;

import java.util.Properties;

public class Main {

    public static void main(String[] args) {

        Properties neuralNetworkProperties = new Properties();

        neuralNetworkProperties.put(Constants.NUMBER_OF_FEATURES, 2);

        neuralNetworkProperties.put(Constants.NUMBER_OF_OUTPUT_NODES, 3);
        neuralNetworkProperties.put(Constants.OUTPUT_ACTIVATION_FUNCTION, "tanh");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYERS, 2);

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "1"), 4);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "1"), "sigmoid");

        neuralNetworkProperties.put(Constants.NUMBER_OF_HIDDEN_LAYER_NODES.replace("{HIDDEN_LAYER}", "2"), 16);
        neuralNetworkProperties.put(Constants.HIDDEN_ACTIVATION_FUNCTION.replace("{HIDDEN_LAYER}", "2"), "relu");



        neuralNetworkProperties.put(Constants.LEARNING_RATE, 0.001);

//        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "MeanSquaredError");
        neuralNetworkProperties.put(Constants.LOSS_FUNCTION, "MeanSquaredError");

        neuralNetworkProperties.put(Constants.BATCH_SIZE, 128);
        neuralNetworkProperties.put(Constants.MAX_EPOCH, 1000000);


        float[][] x = getInputData();
        float[][] y = getLabels();

        Configuration configuration = new Configuration(neuralNetworkProperties);
        try {
            NeuralNetwork neuralNetwork = new NeuralNetwork(configuration);

            neuralNetwork.fetchDataSet(x, y);
            //neuralNetwork.train();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static float[][] getInputData() {
        float[][] x = {{0.1f, 0.3f}, {0.4f, 0.9f}, {0.1f, 0.2f}};
        return x;
    }

    private static float[][] getLabels() {
        float[][] y = {{0.03f, 0.02f, 0.01f}, {0.05f, 0.02f, 0.01f}, {0.07f, 0.02f, 0.01f}};
//        float[][] y = {{0.03f}, {0.05f}, {0.07f}};
        return y;
    }

}
