import java.io.*;
import java.util.*;

public class MultiFeatureLinearRegression {
    static class DataPoint {
        double[] features;
        double label;

        DataPoint(double[] features, double label) {
            this.features = features;
            this.label = label;
        }
    }

    public static void main(String[] args) throws IOException {
        List<DataPoint> data = loadData("src/housing.csv");

        int numFeatures = data.get(0).features.length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];

        normalize(data, means, stds);

        double[] weights = new double[numFeatures]; // initialized to 0
        double bias = 0;
        double lr = 0.01;
        int epochs = 500;

        for (int epoch = 0; epoch <= epochs; epoch++) {
            double[] grads = new double[numFeatures];
            double biasGrad = 0;
            for (DataPoint dp : data) {
                double pred = bias;
                for (int j = 0; j < numFeatures; j++)
                    pred += weights[j] * dp.features[j];

                double error = pred - dp.label;
                for (int j = 0; j < numFeatures; j++)
                    grads[j] += error * dp.features[j];
                biasGrad += error;
            }

            for (int j = 0; j < numFeatures; j++)
                weights[j] -= lr * grads[j] / data.size();
            bias -= lr * biasGrad / data.size();

            if (epoch % 100 == 0)
                System.out.printf("Epoch %d - MSE: %.4f\n", epoch, computeMSE(data, weights, bias));
        }

        // Predict on one input (example: normalized features)
        double[] testInput = normalizeInput(new double[]{1000,2,2,1}, means, stds);
        double pred = bias;
        for (int i = 0; i < numFeatures; i++) pred += weights[i] * testInput[i];
        System.out.println("Predicted price: " + pred);
    }

    static List<DataPoint> loadData(String path) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine(); // header
            
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                double sqft = Double.parseDouble(tokens[0]);          // e.g. 2300
                double bed = Double.parseDouble(tokens[1]);     // Bedrooms
                double bath = Double.parseDouble(tokens[2]);       // Bathrooms
                double laundry = Double.parseDouble(tokens[3]); // 0 or 1 // Laundry room presence
                double price = Double.parseDouble(tokens[4]);         // Price (USD)
                
                data.add(new DataPoint(new double[]{sqft, bed, bath, laundry}, price));
            }
        } // header
        catch (FileNotFoundException e) {
            System.err.println("File not found: " + path);
        } catch (IOException e) {
            System.err.println("Error reading file: " + path);
        } catch(IndexOutOfBoundsException e) {
            System.err.println("Data format error in file: " + path);
        } 
        return data;
    }

    static void normalize(List<DataPoint> data, double[] means, double[] stds) {
        int numFeatures = data.get(0).features.length;
        for (int j = 0; j < numFeatures; j++) {
            for (DataPoint dp : data)
                means[j] += dp.features[j];
            means[j] /= data.size();
        }

        for (int j = 0; j < numFeatures; j++) {
            for (DataPoint dp : data)
                stds[j] += Math.pow(dp.features[j] - means[j], 2);
            stds[j] = Math.sqrt(stds[j] / data.size());
        }

        for (DataPoint dp : data) {
            for (int j = 0; j < numFeatures; j++) {
                if (stds[j] != 0)
                    dp.features[j] = (dp.features[j] - means[j]) / stds[j];
            }
        }
    }

    static double[] normalizeInput(double[] input, double[] means, double[] stds) {
        double[] out = new double[input.length];
        for (int i = 0; i < input.length; i++)
            out[i] = (input[i] - means[i]) / stds[i];
        return out;
    }

    static double computeMSE(List<DataPoint> data, double[] weights, double bias) {
        double sum = 0;
        for (DataPoint dp : data) {
            double pred = bias;
            for (int j = 0; j < weights.length; j++)
                pred += weights[j] * dp.features[j];
            sum += Math.pow(pred - dp.label, 2);
        }
        return sum / data.size();
    }
}
