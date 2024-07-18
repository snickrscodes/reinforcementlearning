package snickrs.cartpole;

import java.util.ArrayList;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Memory {
	ArrayList<INDArray> stateLs = new ArrayList<>();
    ArrayList<Integer> actionLs = new ArrayList<>();
    ArrayList<INDArray> rewardLs = new ArrayList<>();
    ArrayList<INDArray> probsLs = new ArrayList<>();
    ArrayList<Boolean> doneLs = new ArrayList<>();
    public void saveData(INDArray state, INDArray probs, int action, double reward, boolean done) {
		stateLs.add(state);
		probsLs.add(probs);
		actionLs.add(action);
		rewardLs.add(Nd4j.create(new double[] {reward}, new int[] {1, 1}));
		doneLs.add(done);
	}
    public void clear() {
    	stateLs.clear();
    	actionLs.clear();
    	probsLs.clear();
    	rewardLs.clear();
    	doneLs.clear();
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
}

