package snickrs.snakeai;

import java.util.ArrayList;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Game {
	ArrayList<INDArray> stateLs = new ArrayList<>();
    ArrayList<Integer> actionLs = new ArrayList<>();
    ArrayList<Double> rewardLs = new ArrayList<>();
    ArrayList<INDArray> probsLs = new ArrayList<>();
    ArrayList<Boolean> doneLs = new ArrayList<>();
    public void saveData(INDArray state, INDArray probs, int action, double reward, boolean done) {
		stateLs.add(state);
		probsLs.add(probs);
		actionLs.add(action);
		rewardLs.add(reward);
		doneLs.add(done);
	}
    public void clear() {
    	stateLs.clear();
    	actionLs.clear();
    	probsLs.clear();
    	rewardLs.clear();
    	doneLs.clear();
    	System.gc();
    }
    public void print(int idx) {
    	System.out.println("State: ");
    	System.out.println(stateLs.get(idx));
    	System.out.println("Action: " + Integer.toString(actionLs.get(idx)));
    	System.out.println("Reward: ");
    	System.out.println(rewardLs.get(idx));
    	System.out.println("Probs: ");
    	System.out.println(stateLs.get(idx));
    	System.out.println("Done: " + Boolean.toString(doneLs.get(idx)));
    }
    public Game clone() {
    	Game g = new Game();
    	for(int i = 0; i < actionLs.size(); i++) {
    		g.saveData(stateLs.get(i), probsLs.get(i), actionLs.get(i), rewardLs.get(i), doneLs.get(i));
    	}
    	g.stateLs.add(stateLs.get(stateLs.size()-1));
    	return g;
    }
}



