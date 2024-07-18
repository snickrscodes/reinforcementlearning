package snickrs.cartpole;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class Actor {
	SameDiff sd = SameDiff.create();
	int inputs = 4;
	double lr = 0.001;
	int hidden = 512;
	double clip = 0.2;
	int outputs = 2;//2 outputs for actor head and 1 for critic head
	//placeholders
	SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputs); //4 inputs
    SDVariable probs = sd.placeHolder("probs", DataType.FLOAT, -1, outputs);//2 outputs from old state
    SDVariable advantage = sd.placeHolder("advantage", DataType.FLOAT, -1, 1);//1 advantage per state
    SDVariable action = sd.placeHolder("action", DataType.FLOAT, -1, outputs);//2 possible actions per state (one-hot)
    //weights and biases and forward pass
    SDVariable weights_1 = sd.var("weights_1", new XavierInitScheme('c', inputs, hidden), DataType.FLOAT, inputs, hidden);
    SDVariable bias_1 = sd.var("bias_1", DataType.FLOAT, hidden);
    SDVariable activation_1 = sd.nn().leakyRelu("activation_1", sd.mmul(input, weights_1).add(bias_1), 0.01);
    SDVariable weights_2 = sd.var("weights_2", new XavierInitScheme('c', hidden, outputs), DataType.FLOAT, hidden, outputs);
    SDVariable bias_2 = sd.var("bias_2", DataType.FLOAT, outputs);
    SDVariable activation_2 = sd.nn().softmax("activation_2", sd.mmul(activation_1, weights_2).add(bias_2));
    //loss variables for backward pass
	SDVariable logprobs = sd.math().log(activation_2.add(1e-5));
    SDVariable ratio = sd.math().exp(logprobs.sub(sd.math().log(probs.add(1e-5))));
    SDVariable surr1 = ratio.mul(advantage);
    SDVariable clippedRatio = sd.clipByValue(ratio, 1.0-clip, 1.0+clip);
    SDVariable surr2 = clippedRatio.mul(advantage);
    SDVariable loss = sd.min(surr1, surr2).mul(action).neg();
	public Actor(double lr, double clip) {
		this.lr = lr;
		this.clip = clip;
		TrainingConfig config = new TrainingConfig.Builder()
	            .l2(1e-4)
	            .updater(new Adam(lr))
	            .dataSetFeatureMapping("input")
	            .dataSetLabelMapping("probs", "advantage", "action") 
	            .build();
	    sd.setTrainingConfig(config);
	}
	public INDArray output(INDArray state) {
		sd.associateArrayWithVariable(state, input);
		return activation_2.eval();
	}
}
