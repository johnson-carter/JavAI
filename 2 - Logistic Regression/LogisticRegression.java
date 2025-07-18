public class LogisticRegression {
    private double theta0 = 0, theta1 = 0;
    private double learningRate = 0.1;
    private int iterations = 1000;

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public void train(double[] x, int[] y) {
        int m = x.length;
        for (int iter = 0; iter < iterations; iter++) {
            double grad0 = 0, grad1 = 0;
            for (int i = 0; i < m; i++) {
                double z = theta0 + theta1 * x[i];
                double prediction = sigmoid(z);
                double error = prediction - y[i];
                grad0 += error;
                grad1 += error * x[i];
            }
            grad0 /= m;
            grad1 /= m;
            theta0 -= learningRate * grad0;
            theta1 -= learningRate * grad1;
        }
    }

    public double predictProbability(double x) {
        double z = theta0 + theta1 * x;
        return sigmoid(z);
    }

    public int predictClass(double x) {
        return predictProbability(x) >= 0.5 ? 1 : 0;
    }

    public static void main(String[] args) {
        // Simple example dataset: x = feature, y = class label (0 or 1)
        double[] x = {1, 2, 3, 4, 5};
        int[] y = {0, 0, 0, 1, 1};  // Class changes between 3 and 4

        LogisticRegression model = new LogisticRegression();
        model.train(x, y);

        System.out.println("Prediction probability for 2.5: " + model.predictProbability(2.5));
        System.out.println("Prediction class for 2.5: " + model.predictClass(2.5));
        System.out.println("Prediction probability for 4.5: " + model.predictProbability(4.5));
        System.out.println("Prediction class for 4.5: " + model.predictClass(4.5));
    }
}
