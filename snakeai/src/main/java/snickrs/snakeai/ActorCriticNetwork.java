package snickrs.snakeai;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActorCriticNetwork {
	public ComputationGraph agent;
//	public ParallelInference pi;
//	public Game[] games = new Game[SnakeAI.NUM_SNAKES];
	public ActorCriticNetwork() {
		makeModel();
//		for(int i = 0; i < games.length; i++) {
//			games[i] = new Game();
//		}
	}
//	//use this for loading models
//	public ActorCriticNetwork(String path, int index) {
//		String s = Integer.toString(index);
//		try {
//			agent = ComputationGraph.load(new File(path+name+s+".dl4j"), true);
//		} catch (IOException e) {
//			System.out.println("ur L computer doesnt have an existing file");
//		}
////		for(int i = 0; i < games.length; i++) {
////			games[i] = new Game();
////		}
//	}
//	public void save(String path, int index) {
//		String s = Integer.toString(index);
//		try {
//		  agent.save(new File(path+name+s+".dl4j"));
//		} catch (IOException e) {
//			System.out.println("ur L computer couldnt save the files");
//		}
//	}
//	public void learn(int i) {
//		fit(games[i]);
//		System.gc();
//	}
//	public void learn() {
////		fit();
//		for(int i = 0; i < games.length; i++) {
//			fit(games[i]);
//		}
//		System.gc();
//	}
    public void fit(Game game) {
    	INDArray states = game.stateLs.get(0);
    	INDArray[] advantage = calcAdvantage(game);
    	//one hot actions
    	INDArray actions = Nd4j.zeros(game.actionLs.size(), SnakeAI.actions);
    	for (int i = 0; i < game.actionLs.size(); i++) {
    		actions.putScalar(i, game.actionLs.get(i), 1);
    		if (i != 0) {
    			states = Nd4j.concat(0, states, game.stateLs.get(i));
    		}
    	}
    	INDArray temp = actions.mul(advantage[0]);
    	MultiDataSet set = new MultiDataSet();
    	set.setFeatures(new INDArray[] {states});
    	set.setLabels(new INDArray[] {temp, advantage[1]});
    	agent.fit(set);
    	game.clear();
    }
    public INDArray[] getSet(Game game) {
    	INDArray states = game.stateLs.get(0);
    	INDArray[] advantage = calcAdvantage(game);
    	//one hot actions
    	INDArray actions = Nd4j.zeros(game.actionLs.size(), SnakeAI.actions);
    	for (int i = 0; i < game.actionLs.size(); i++) {
    		actions.putScalar(i, game.actionLs.get(i), 1);
    		if (i != 0) {
    			states = Nd4j.concat(0, states, game.stateLs.get(i));
    		}
    	}
    	INDArray temp = actions.mul(advantage[0]);
    	return new INDArray[] {states, temp, advantage[1]};
    }
    public void fit(Game[] games) {
    	INDArray[] data = getSet(games[0]);
    	for(int i = 1; i < games.length; i++) {
    		INDArray[] newdata = getSet(games[i]);
    		data[0] = Nd4j.concat(0, data[0], newdata[0]);
    		data[1] = Nd4j.concat(0, data[1], newdata[1]);
    		data[2] = Nd4j.concat(0, data[2], newdata[2]);
    	}
    	
    	MultiDataSet set = new MultiDataSet();
    	set.setFeatures(new INDArray[] {data[0]});
    	set.setLabels(new INDArray[] {data[1], data[2]});
    	agent.fit(set);
    	System.gc();
    }

public INDArray[] calcAdvantage(Game game) {
	INDArray advantage = Nd4j.zeros(game.actionLs.size(),1);
	INDArray td = Nd4j.zeros(game.actionLs.size(),1);
	for (int i = 0; i < game.actionLs.size(); i++) {
		if (i == game.actionLs.size()-1) {
			advantage.putScalar(i, game.rewardLs.get(i));
			td.putScalar(i, game.rewardLs.get(i));
			continue;
		}
		INDArray predQ = agent.output(game.stateLs.get(i))[1];
		INDArray futQ = agent.output(game.stateLs.get(i+1))[1];
		double val = game.rewardLs.get(i) + SnakeAI.gamma * futQ.getDouble(0) * (game.doneLs.get(i) ? 0 : 1);
		td.putScalar(i, val);
		advantage.putScalar(i, val-predQ.getDouble(0));
	}
	return new INDArray[] {advantage, td};
}

public static int sample(INDArray logits) {
	INDArray u = Nd4j.rand(logits.shape());
	INDArray l = Transforms.log(u, true).neg();
	INDArray l2 = Transforms.log(l, true);
	INDArray s = logits.sub(l2);
	return s.argMax(-1).getInt(0);
}

public INDArray calcTD(Game game) {
	INDArray val = Nd4j.zeros(game.rewardLs.size(), 1);
	double sum = 0;
	for (int i = 0; i < game.rewardLs.size(); i++) {
		sum = game.rewardLs.get(i) + SnakeAI.gamma * sum * (game.doneLs.get(i) ? 0 : 1);
		val.putScalar(game.rewardLs.size()-1-i, sum);
	}
	return val;
}
	public void makeModel() {
//		Nd4j.getRandom().setSeed(SnakeAI.seed);
		ComputationGraphConfiguration conf;
		if(SnakeAI.convnet) {
			//old net for 10x10
//			conf = new NeuralNetConfiguration.Builder()
//				.seed(SnakeAI.seed)
//				.updater(new Adam(SnakeAI.lr))
//				.l2(SnakeAI.l2)
//				.weightInit(WeightInit.XAVIER)
//				.biasInit(0)
//				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//				.convolutionMode(ConvolutionMode.Same)
//				.graphBuilder()
//				.addInputs("input")
//				.addLayer("conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nIn(SnakeAI.channels).nOut(64).activation(Activation.RELU).build(), "input")
//		        .addLayer("conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(), "conv1")
//		        .addLayer("conv3", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(), "conv2")
//		        .addLayer("conv4", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(), "conv3")
//		        .addLayer("aconv1", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(2).activation(Activation.RELU).build(), "conv4")
//				.addLayer("out1", new OutputLayer.Builder().nOut(SnakeAI.actions).activation(Activation.SOFTMAX).lossFunction(new ActorCriticLoss()).build(), "aconv1")
//				.addLayer("cconv1", new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(1).activation(Activation.RELU).build(), "conv4")
//				.addLayer("L1", new DenseLayer.Builder().nOut(64).activation(Activation.RELU).build(), "cconv1")
//				.addLayer("out2", new OutputLayer.Builder().nOut(1).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(), "L1")
//				.setOutputs("out1", "out2")
//				.setInputTypes(InputType.convolutional(SnakeAI.GAME_SIZE+2, SnakeAI.GAME_SIZE+2, SnakeAI.channels))
//				.build();
			//net for 6x6
			conf = new NeuralNetConfiguration.Builder()
					.seed(SnakeAI.seed)
					.updater(new Adam(SnakeAI.lr))
					.l2(SnakeAI.l2)
					.weightInit(WeightInit.XAVIER)
					.convolutionMode(ConvolutionMode.Same)
					.biasInit(0)
					.graphBuilder()
					.addInputs("input")
					.addLayer("conv1", new ConvolutionLayer.Builder(3, 3).stride(2, 2).nIn(SnakeAI.observation_space[2]).nOut(32).activation(Activation.RELU).build(), "input")
			        .addLayer("conv2", new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(64).activation(Activation.RELU).build(), "conv1")
			        .addLayer("conv3", new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(64).activation(Activation.RELU).build(), "conv2")
					.addLayer("AL1", new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build(), "conv3")
					.addLayer("out1", new OutputLayer.Builder().nOut(SnakeAI.actions).activation(Activation.SOFTMAX).lossFunction(new PolicyLoss()).build(), "AL1")
					.addLayer("CL1", new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build(), "conv3")
					.addLayer("out2", new OutputLayer.Builder().nOut(1).activation(Activation.IDENTITY).lossFunction(new ValueLoss()).build(), "CL1")
					.setOutputs("out1", "out2")
					.setInputTypes(InputType.convolutional(SnakeAI.observation_space[1], SnakeAI.observation_space[0], SnakeAI.observation_space[2]))
					.build();
		} else {
			conf = new NeuralNetConfiguration.Builder()
					.seed(SnakeAI.seed)
					.updater(new Adam(SnakeAI.lr))
					.l2(SnakeAI.l2)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0)
					.graphBuilder()
					.addInputs("input")
					.addLayer("L1", new DenseLayer.Builder().nIn(11).nOut(256).activation(Activation.RELU).build(), "input")
					.addLayer("out1", new OutputLayer.Builder().nIn(256).nOut(SnakeAI.actions).activation(Activation.SOFTMAX).lossFunction(new PolicyLoss()).build(), "L1")
					.addLayer("out2", new OutputLayer.Builder().nIn(256).nOut(1).activation(Activation.IDENTITY).lossFunction(new ValueLoss()).build(), "L1")
					.setOutputs("out1", "out2")
					.build();
		}
		agent = new ComputationGraph(conf);
		agent.init();
//		pi = new ParallelInference.Builder(agent).inferenceMode(InferenceMode.INPLACE).queueLimit(256).build();
	}
	
}

