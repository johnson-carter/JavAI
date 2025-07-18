import java.io.*;
import java.util.*;

abstract class Model {
    double[] weights;
    double learningRate = 0.01;

    public void initialize(int featureCount) {
        weights = new double[featureCount + 1]; // +1 for bias
        for (int i = 0; i < weights.length; i++)
            weights[i] = Math.random() - 0.5;
    }

    public abstract double predict(double[] input);

    public abstract void train(double[][] X, double[] y, int epochs);

    protected double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];
        return sum;
    }
}

class RegressionModel extends Model {
    @Override
    public double predict(double[] input) {
        double sum = weights[0]; // bias
        for (int i = 0; i < input.length; i++)
            sum += weights[i + 1] * input[i];
        return sum;
    }

    @Override
    public void train(double[][] X, double[] y, int epochs) {
        int n = X.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                double[] xi = X[i];
                double predicted = predict(xi);
                double error = predicted - y[i];

                weights[0] -= learningRate * error; // bias update
                for (int j = 0; j < xi.length; j++)
                    weights[j + 1] -= learningRate * error * xi[j];
            }

            if (epoch % 100 == 0) {
                double mse = 0;
                for (int i = 0; i < n; i++) {
                    double error = predict(X[i]) - y[i];
                    mse += error * error;
                }
                System.out.printf("Epoch %d  MSE=%.4f\n", epoch, mse / n);
            }
        }
    }
}

class LogisticModel extends Model {
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double predict(double[] input) {
        double sum = weights[0];
        for (int i = 0; i < input.length; i++)
            sum += weights[i + 1] * input[i];
        return sigmoid(sum);
    }

    @Override
    public void train(double[][] X, double[] y, int epochs) {
        int n = X.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                double[] xi = X[i];
                double predicted = predict(xi);
                double error = predicted - y[i];

                weights[0] -= learningRate * error;
                for (int j = 0; j < xi.length; j++)
                    weights[j + 1] -= learningRate * error * xi[j];
            }

            if (epoch % 100 == 0) {
                double loss = 0;
                for (int i = 0; i < n; i++) {
                    double p = predict(X[i]);
                    loss += -y[i] * Math.log(p + 1e-10) - (1 - y[i]) * Math.log(1 - p + 1e-10);
                }
                System.out.printf("Epoch %d  LogLoss=%.4f\n", epoch, loss / n);
            }
        }
    }
}

public class UnifiedML {
    public static void main(String[] args) {
        // ✅ Hardcoded input
        String filePath = "income_data.csv";
        String mode = "regression"; // or "classification"

        try {
            List<double[]> data = new ArrayList<>();
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String header = br.readLine(); // discard header
            String line;
            while ((line = br.readLine()) != null && !line.isBlank()) {
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++)
                    row[i] = Double.parseDouble(parts[i]);
                data.add(row);
            }
            br.close();

            int featureCount = data.get(0).length - 1;
            double[][] X = new double[data.size()][featureCount];
            double[] y = new double[data.size()];
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < featureCount; j++)
                    X[i][j] = data.get(i)[j];
                y[i] = data.get(i)[featureCount];
            }

            // ✅ Normalize features
            normalize(X);

            // ✅ Choose model
            Model model = mode.equalsIgnoreCase("classification")
                ? new LogisticModel()
                : new RegressionModel();

            model.initialize(featureCount);
            model.train(X, y, 1000);

            // ✅ Sample prediction
            double[] testInput = new double[] {13, 19, 1, 0}; // education, age, experience, isUrban
            normalizeRow(testInput, X); // reuse normalization ranges
            double prediction = model.predict(testInput);
            System.out.printf("Prediction for input %s = %.4f\n", Arrays.toString(testInput), prediction);
            System.out.println("Weights: " + Arrays.toString(model.weights));
        } catch (IOException e) {
            System.err.println("File error: " + e.getMessage());
        }
    }

    private static void normalize(double[][] X) {
        int features = X[0].length;
        for (int j = 0; j < features; j++) {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            for (double[] row : X) {
                min = Math.min(min, row[j]);
                max = Math.max(max, row[j]);
            }
            for (double[] row : X) {
                if (max - min != 0)
                    row[j] = (row[j] - min) / (max - min);
            }
        }
    }

    private static void normalizeRow(double[] input, double[][] X) {
        int features = input.length;
        for (int j = 0; j < features; j++) {
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            for (double[] row : X) {
                min = Math.min(min, row[j]);
                max = Math.max(max, row[j]);
            }
            if (max - min != 0)
                input[j] = (input[j] - min) / (max - min);
        }
    }
}
