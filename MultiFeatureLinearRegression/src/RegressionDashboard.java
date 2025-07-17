import java.awt.*;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import javax.swing.*; // Added for array printing

public class RegressionDashboard extends JFrame {

    private MultiFeatureLinearRegression model; // Instance of your model
    private double[] currentWeights;
    private double currentBias;
    private double[] currentMeans;
    private double[] currentStds;
    private List<MultiFeatureLinearRegression.DataPoint> currentData; // Store current data

    // UI Components
    private JTextField sqftField, bedField, bathField, laundryField;
    private JButton predictButton, addDataButton, viewWeightsButton, retrainButton;
    private JTextArea outputArea;

    private static final String DATA_FILE = "src/housing.csv"; // Your data file

    public RegressionDashboard() {
        super("Multi-Feature Linear Regression Dashboard");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new BorderLayout());

        // --- Output Area ---
        outputArea = new JTextArea(); // Initialize before loadAndTrainModel()
        outputArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(outputArea);
        add(scrollPane, BorderLayout.SOUTH);

        // Initialize the model and load initial data
        model = new MultiFeatureLinearRegression();
        loadAndTrainModel(); // Initial load and train

        // --- Input Panel ---
        JPanel inputPanel = new JPanel(new GridLayout(5, 2, 10, 10));
        inputPanel.setBorder(BorderFactory.createTitledBorder("Prediction Input / New Data"));

        inputPanel.add(new JLabel("Square Footage:"));
        sqftField = new JTextField();
        inputPanel.add(sqftField);

        inputPanel.add(new JLabel("Bedrooms:"));
        bedField = new JTextField();
        inputPanel.add(bedField);

        inputPanel.add(new JLabel("Bathrooms:"));
        bathField = new JTextField();
        inputPanel.add(bathField);

        inputPanel.add(new JLabel("Laundry (0 or 1):"));
        laundryField = new JTextField();
        inputPanel.add(laundryField);

        add(inputPanel, BorderLayout.NORTH);

        // --- Button Panel ---
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 10));
        predictButton = new JButton("Predict Price");
        addDataButton = new JButton("Add New Data & Retrain");
        viewWeightsButton = new JButton("View Model Weights");
        retrainButton = new JButton("Retrain Model"); // New button for explicit retraining

        buttonPanel.add(predictButton);
        buttonPanel.add(addDataButton);
        buttonPanel.add(viewWeightsButton);
        buttonPanel.add(retrainButton);

        add(buttonPanel, BorderLayout.CENTER);

        // --- Action Listeners ---
        predictButton.addActionListener(e -> predictPrice());
        addDataButton.addActionListener(e -> addNewData());
        viewWeightsButton.addActionListener(e -> displayWeights());
        retrainButton.addActionListener(e -> retrainModelExplicitly());

        setVisible(true);
    }

    private void loadAndTrainModel() {
        try {
            currentData = model.loadData(DATA_FILE);
            if (currentData.isEmpty()) {
                outputArea.append("Error: No data loaded from " + DATA_FILE + "\n");
                return;
            }

            int numFeatures = currentData.get(0).features.length;
            currentMeans = new double[numFeatures];
            currentStds = new double[numFeatures];
            model.normalize(currentData, currentMeans, currentStds);

            currentWeights = new double[numFeatures]; // Initialize weights for training
            Arrays.fill(currentWeights, 0.0); // Ensure weights are reset
            currentBias = 0.0;

            // Retrain the model
            int epochs = 500;
            double lr = 0.01;

            outputArea.append("Starting model training...\n");
            for (int epoch = 0; epoch <= epochs; epoch++) {
                double[] grads = new double[numFeatures];
                double biasGrad = 0;
                for (MultiFeatureLinearRegression.DataPoint dp : currentData) {
                    double pred = currentBias;
                    for (int j = 0; j < numFeatures; j++)
                        pred += currentWeights[j] * dp.features[j];

                    double error = pred - dp.label;
                    for (int j = 0; j < numFeatures; j++)
                        grads[j] += error * dp.features[j];
                    biasGrad += error;
                }

                for (int j = 0; j < numFeatures; j++)
                    currentWeights[j] -= lr * grads[j] / currentData.size();
                currentBias -= lr * biasGrad / currentData.size();

                if (epoch % 100 == 0) {
                    outputArea.append(String.format("Epoch %d - MSE: %.4f\n", epoch, model.computeMSE(currentData, currentWeights, currentBias)));
                }
            }
            outputArea.append("Model training complete.\n");

        } catch (IOException e) {
            outputArea.append("Error loading or processing data: " + e.getMessage() + "\n");
        } catch (Exception e) {
            outputArea.append("An unexpected error occurred during model loading/training: " + e.getMessage() + "\n");
            e.printStackTrace();
        }
    }

    private void predictPrice() {
        try {
            double sqft = Double.parseDouble(sqftField.getText());
            double bed = Double.parseDouble(bedField.getText());
            double bath = Double.parseDouble(bathField.getText());
            double laundry = Double.parseDouble(laundryField.getText());

            double[] inputFeatures = new double[]{sqft, bed, bath, laundry};
            double[] normalizedInput = model.normalizeInput(inputFeatures, currentMeans, currentStds);

            double predictedPrice = currentBias;
            for (int i = 0; i < currentWeights.length; i++) {
                predictedPrice += currentWeights[i] * normalizedInput[i];
            }
            outputArea.append(String.format("Predicted Price for [%.0f, %.0f, %.0f, %.0f]: $%.2f\n", sqft, bed, bath, laundry, predictedPrice));

        } catch (NumberFormatException ex) {
            outputArea.append("Error: Please enter valid numerical values for all fields.\n");
        } catch (NullPointerException ex) {
            outputArea.append("Error: Model not yet trained or data not loaded. Please ensure the data file is present and valid.\n");
        } catch (Exception ex) {
            outputArea.append("An error occurred during prediction: " + ex.getMessage() + "\n");
            ex.printStackTrace();
        }
    }

    private void addNewData() {
        try {
            double sqft = Double.parseDouble(sqftField.getText());
            double bed = Double.parseDouble(bedField.getText());
            double bath = Double.parseDouble(bathField.getText());
            double laundry = Double.parseDouble(laundryField.getText());
            
            // For adding new data, we assume a placeholder for price, 
            // as we don't have a new label input field in this UI for it.
            // In a real scenario, you'd likely have a separate input for the actual price
            // of the new data point you're adding to retrain on.
            // For simplicity, let's just use the current prediction as a "placeholder" or assume it's for future prediction
            // if we were to only add features.
            // However, for retraining, we need a label. Let's assume the user intends to add a *complete* new data point.
            // For this example, let's make a dummy price for the new data point for demonstration purposes,
            // or better, if the user is adding new *training* data, they should provide the label.
            // To be practical for adding "training data", we'd need another input field for the actual price (label).
            // For now, let's assume the intent is to add to the CSV for future retraining.
            
            String newLine = String.format("%.0f,%.0f,%.0f,%.0f,0.0\n", sqft, bed, bath, laundry); // Placeholder 0.0 for price, user should manually edit or provide.
                                                                                                    //  A better UI would have a price input for new training data.

            // To actually add new TRAINING data, the user MUST input a price.
            // For simplicity in this example, I'll prompt the user for the price.
            String priceStr = JOptionPane.showInputDialog(this, "Enter the actual price for the new data point:");
            if (priceStr == null || priceStr.trim().isEmpty()) {
                outputArea.append("New data not added: Price not provided.\n");
                return;
            }
            double price = Double.parseDouble(priceStr);
            newLine = String.format("%.0f,%.0f,%.0f,%.0f,%.2f\n", sqft, bed, bath, laundry, price);

            try (FileWriter fw = new FileWriter(DATA_FILE, true); // true for append mode
                 BufferedWriter bw = new BufferedWriter(fw);
                 PrintWriter out = new PrintWriter(bw)) {
                out.print(newLine); // Use print to avoid an extra newline if BufferedWriter adds one
                outputArea.append("Added new data point to " + DATA_FILE + ": " + newLine.trim() + "\n");
                outputArea.append("Retraining model with new data...\n");
                loadAndTrainModel(); // Retrain the model with the updated data
            } catch (IOException e) {
                outputArea.append("Error writing to file: " + e.getMessage() + "\n");
            }

        } catch (NumberFormatException ex) {
            outputArea.append("Error: Please enter valid numerical values for all fields.\n");
        } catch (Exception ex) {
            outputArea.append("An error occurred while adding new data: " + ex.getMessage() + "\n");
            ex.printStackTrace();
        }
    }

    private void displayWeights() {
        if (currentWeights == null || currentMeans == null || currentStds == null) {
            outputArea.append("Model not yet trained. Please ensure data is loaded and trained.\n");
            return;
        }
        outputArea.append("--- Model Weights and Bias ---\n");
        outputArea.append("Weights (normalized features): " + Arrays.toString(currentWeights) + "\n");
        outputArea.append("Bias: " + currentBias + "\n");
        outputArea.append("Feature Means: " + Arrays.toString(currentMeans) + "\n");
        outputArea.append("Feature Standard Deviations: " + Arrays.toString(currentStds) + "\n");
        outputArea.append("------------------------------\n");
    }

    private void retrainModelExplicitly() {
        outputArea.append("Initiating explicit model retraining...\n");
        loadAndTrainModel();
        outputArea.append("Explicit retraining complete.\n");
    }

    public static void main(String[] args) {
        // Create a dummy housing.csv file if it doesn't exist for testing
        File dataFile = new File("src/housing.csv");
        if (!dataFile.exists()) {
            try {
                dataFile.getParentFile().mkdirs(); // Ensure parent directories exist
                PrintWriter writer = new PrintWriter(dataFile);
                writer.println("sqft,bed,bath,laundry,price");
                writer.println("1500,3,2,1,250000");
                writer.println("2000,4,3,1,350000");
                writer.println("1200,2,1,0,180000");
                writer.println("1800,3,2,0,300000");
                writer.println("2200,4,3,1,400000");
                writer.close();
                System.out.println("Created dummy housing.csv for testing.");
            } catch (IOException e) {
                System.err.println("Could not create dummy housing.csv: " + e.getMessage());
            }
        }

        SwingUtilities.invokeLater(RegressionDashboard::new);
    }
}