import java.io.*;
import java.util.*;

// === Interface for models ===
interface Model {
    void train(double[] x, double[] y);
    double predict(double x);
}

// === Linear regression for continuous features ===
class LinearRegression implements Model {
    double weight = 0, bias = 0, lr = 0.01;

    public void train(double[] x, double[] y) {
        for (int epoch = 0; epoch < 500; epoch++) {
            double dw = 0, db = 0;
            for (int i = 0; i < x.length; i++) {
                double yhat = weight * x[i] + bias;
                dw += (yhat - y[i]) * x[i];
                db += (yhat - y[i]);
            }
            dw /= x.length;
            db /= x.length;
            weight -= lr * dw;
            bias -= lr * db;
        }
    }

    public double predict(double x) {
        return weight * x + bias;
    }
}

// === Logistic regression for binary features ===
class LogisticRegression implements Model {
    double weight = 0, bias = 0, lr = 0.01;

    public void train(double[] x, double[] y) {
        for (int epoch = 0; epoch < 500; epoch++) {
            double dw = 0, db = 0;
            for (int i = 0; i < x.length; i++) {
                double yhat = sigmoid(weight * x[i] + bias);
                dw += (yhat - y[i]) * x[i];
                db += (yhat - y[i]);
            }
            dw /= x.length;
            db /= x.length;
            weight -= lr * dw;
            bias -= lr * db;
        }
    }

    public double predict(double x) {
        return sigmoid(weight * x + bias);
    }

    private double sigmoid(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }
}

// === Feature pipeline ===
class FeaturePipeline {
    List<Model> models = new ArrayList<>();
    boolean[] isBinary; // true = use logistic, false = use linear

    public FeaturePipeline(boolean[] isBinary) {
        this.isBinary = isBinary;
    }

    public void train(double[][] X, double[] y) {
        for (int i = 0; i < X[0].length; i++) {
            double[] col = getColumn(X, i);
            Model model = isBinary[i] ? new LogisticRegression() : new LinearRegression();
            model.train(col, y);
            models.add(model);
        }
    }

    public double predict(double[] x) {
        double total = 0;
        for (int i = 0; i < x.length; i++) {
            total += models.get(i).predict(x[i]);
        }
        return total / x.length; // mean of submodel predictions
    }

    private double[] getColumn(double[][] X, int index) {
        double[] col = new double[X.length];
        for (int i = 0; i < X.length; i++)
            col[i] = X[i][index];
        return col;
    }
}

// === Main class with CSV parsing and example usage ===
public class FeaturePipelineMain {
    public static void main(String[] args) throws IOException {
        String filename = "src/data.csv"; // your input file
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        List<double[]> Xlist = new ArrayList<>();
        List<Double> ylist = new ArrayList<>();

        // Assume CSV format: feat1,feat2,...,label
        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split(",");
            double[] x = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                x[i] = Double.parseDouble(parts[i]);
            }
            double y = Double.parseDouble(parts[parts.length - 1]);
            Xlist.add(x);
            ylist.add(y);
        }
        reader.close();

        double[][] X = Xlist.toArray(new double[0][]);
        double[] y = ylist.stream().mapToDouble(d -> d).toArray();

        // Manually define which features are binary (true = binary/logistic)
        boolean[] isBinary = new boolean[] {true, false, false}; // for 3 features

        FeaturePipeline pipeline = new FeaturePipeline(isBinary);
        pipeline.train(X, y);

        double[] testInput = {1, 2.5, 3.0}; // example input
        System.out.println("Prediction: " + pipeline.predict(testInput));
    }
}
