import java.lang.Math;
import java.util.Random;
import java.util.ArrayList;

public class NeuralNetwork{
    private int hiddenNodes;
    private int outputNodes;
    private ArrayList<Input> trainingData;
    
    void main(){
        // Lets train a neural network to predict the cosine of the input double
        trainingData = loadTrainingData();
        int inputNodes = trainingData.size();
        
    }

    public ArrayList<Input> loadTrainingData(){
        // Various cosine values
        ArrayList<Input> trainingData = new ArrayList<Input>();
        trainingData.add(new Input(0.0, 1));
        trainingData.add(new Input(Math.PI/2, 0));
        trainingData.add(new Input(Math.PI, -1));
        trainingData.add(new Input(3*Math.PI/2, 0));
        trainingData.add(new Input(2*Math.PI, 1));

        trainingData.add(new Input(Math.PI/3, 0.5));
        trainingData.add(new Input(2*Math.PI/3, -0.5));
        trainingData.add(new Input(4*Math.PI/3, -0.5));
        trainingData.add(new Input(5*Math.PI/3, 0.5));

        trainingData.add(new Input(Math.PI/4, Math.sqrt(2)/2));
        trainingData.add(new Input(3*Math.PI/4, -Math.sqrt(2)/2));
        trainingData.add(new Input(5*Math.PI/4, -Math.sqrt(2)/2));
        trainingData.add(new Input(7*Math.PI/4, Math.sqrt(2)/2));

        trainingData.add(new Input(.5, 0.88));
        trainingData.add(new Input(2, -.42));
        trainingData.add(new Input(8*Math.PI, 1));
        trainingData.add(new Input(10, -0.84));
        return trainingData;
    }

}
