package snickrs.cartpole;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.weightinit.impl.XavierInitScheme;

class Critic {
	SameDiff sd = SameDiff.create();
	double lr = 0.001;
	double lossScl = 0.5;
	Adam optimizer = new Adam(lr);
	int inputs = 4;
	int hidden = 512;
	int outputs = 1;//2 outputs for actor head and 1 for critic head
	//placeholders
	SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, inputs); //4 inputs
    SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, outputs);//1 outputs from old state
    //weights and biases and forward pass
    SDVariable weights_1 = sd.var("weights_1", new XavierInitScheme('c', inputs, hidden), DataType.FLOAT, inputs, hidden);
    SDVariable bias_1 = sd.var("bias_1", DataType.FLOAT, hidden);
    SDVariable activation_1 = sd.nn().leakyRelu("activation_1", sd.mmul(input, weights_1).add(bias_1), 0.01);
    SDVariable weights_2 = sd.var("weights_2", new XavierInitScheme('c', hidden, outputs), DataType.FLOAT, hidden, outputs);
    SDVariable bias_2 = sd.var("bias_2", DataType.FLOAT, outputs);
    SDVariable activation_2 = sd.mmul(activation_1, weights_2).add("activation_2", bias_2);
    //loss variables for backward pass (using MSE here)
    SDVariable loss = sd.math().square(labels.sub(activation_2)).mul(lossScl).mean();
	public Critic(double lr) {
		this.lr = lr;
	    TrainingConfig config = new TrainingConfig.Builder()
	            .l2(1e-4)
	            .updater(new Adam(lr))
	            .dataSetFeatureMapping("input")
	            .dataSetLabelMapping("labels") 
	            .build();
	    sd.setTrainingConfig(config);
	}
	public INDArray output(INDArray state) {
		sd.associateArrayWithVariable(state, input);
		return activation_2.eval();
	}
}
