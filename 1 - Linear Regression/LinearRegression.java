public class LinearRegression {

    private double theta0 = 0, theta1 = 0;  // Model parameters
    private double learningRate = 0.03;     // Rate of gradient descent
    private int iterations = 1000;        // Number of iterations for training

    public void train(double[] x, double[] y) {
        int m = x.length;
        for (int iter = 0; iter < iterations; iter++) {
            double grad0 = 0, grad1 = 0;
            for (int i = 0; i < m; i++) {
                double prediction = theta0 + theta1 * x[i];
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

    public double predict(double x) {
        return theta0 + theta1 * x;
    }

    public static void main(String[] args) {
        //Here is a simple example to train the model with. y = 2x (Perfect Linear)
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};  

        //Here is where we create the model and train it
        LinearRegression model = new LinearRegression();
        model.train(x, y); // The "machine learning" step

        System.out.println("Learned theta0: " + model.theta0);
        System.out.println("Learned theta1: " + model.theta1);
        System.out.println("Prediction for 7: " + model.predict(7));
    }
}
