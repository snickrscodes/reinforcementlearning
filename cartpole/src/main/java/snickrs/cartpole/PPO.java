package snickrs.cartpole;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class PPO {
	public Actor actor;
	public Critic critic;
	public int epochs = 2;
	public double lambda = 0.95;
	public double gamma = 0.99;
	public Random random = new Random(0);
	public Memory game = new Memory();
	public PPO(double lr, double clip, double gamma, double lambda, int epochs) {
		this.lambda = lambda;
		this.gamma = gamma;
		this.epochs = epochs;
		actor = new Actor(lr, clip);
		critic = new Critic(lr);
	}
	public void load(String path) throws IOException {
		actor.sd = SameDiff.fromFlatFile(new File(path+"actor.fb"), true);
		critic.sd = SameDiff.fromFlatFile(new File(path+"critic.fb"), true);
	}
	public void save(String path) throws IOException {
		actor.sd.asFlatFile(new File(path+"actor.fb"), true);
		critic.sd.asFlatFile(new File(path+"critic.fb"), true);
	}
	public void learn() {
		fit(game);
	}
    public void fit(Memory game) {
    	INDArray states = game.stateLs.get(0);
    	ArrayList<INDArray> list = calcGAE(game);
    	INDArray advantage = list.get(0);
    	INDArray returns = list.get(1);
    	INDArray doneArr = Nd4j.zeros(game.doneLs.size(), 1);
    	INDArray probs = game.probsLs.get(0);
    	INDArray action = Nd4j.zeros(game.actionLs.size(), 2);
    	for (int i = 0; i < game.actionLs.size(); i++) {
    		action.putScalar(i, game.actionLs.get(i), 1);
    		if (game.doneLs.get(i)) doneArr.putScalar(i, 1);
    		if (i != 0) {
    			states = Nd4j.concat(0, states, game.stateLs.get(i));
    			probs = Nd4j.concat(0, probs, game.probsLs.get(i));
    		}
    	}
    	DataSet crit = new DataSet();
    	MultiDataSet act = new MultiDataSet();
    	crit.setFeatures(states);
    	crit.setLabels(returns);
    	act.setFeatures(new INDArray[] {states.dup()});
    	act.setLabels(new INDArray[] {probs, advantage, action});
    	critic.sd.fit(crit);
    	actor.sd.fit(act);
    	game.clear();
    }

public ArrayList<INDArray> calcGAE(Memory game) {
	INDArray returns = Nd4j.zeros(game.actionLs.size(), 1);
	INDArray advantage;
	INDArray values = Nd4j.zeros(game.actionLs.size(), 1);
	double gae = 0.0;
	for (int i = game.actionLs.size()-1; i >= 0; i--) {
		INDArray predQ = critic.output(game.stateLs.get(i));
		values.put(i, predQ);
		INDArray futQ = critic.output(game.stateLs.get(i+1));
		int mask = game.doneLs.get(i) ? 0 : 1; //no reward for terminal state
		double delta = game.rewardLs.get(i).getDouble(0) + gamma * futQ.getDouble(0) * mask - predQ.getDouble(0);
		gae = delta + gamma * lambda * mask * gae;
        returns.put(i, predQ.addi(gae));
	}
	ArrayList<INDArray> vals = new ArrayList<>();
	advantage = returns.sub(values);
	vals.add(advantage);
	vals.add(returns);
	return vals;
}
public INDArray reverse(INDArray arr) {
	INDArray temp = Nd4j.zeros(arr.rows(), 1);
	for(int i = 0; i < arr.rows(); i++) {
		temp.put(arr.rows()-1-i, arr.getRow(i));
	}
	return temp;
}
	public int action(INDArray state) {
		return sample(actor.output(state));
	}
	public INDArray output(INDArray state) {
		return actor.output(state);
	}
	int sample(INDArray arr) {
		  double[] probs = arr.data().asDouble();
		  double x = random.nextDouble()*sum(probs);
		  for (int i = 0; i < probs.length; ++i) {
		    x -= probs[i];
		    if (x <= 0) {
		      return i;
		    }
		  }
		  return probs.length-1;
		}
	double sum(double[] arr) {
		  double total = 0;
		  for (int i = 0; i < arr.length; i++) {
		    total += arr[i];
		  }
		  return total;
		}
}
