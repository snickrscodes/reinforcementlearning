package snickrs.snakeai;

import java.util.Random;
import org.nd4j.linalg.factory.Nd4j;

public class SnakeAI {
public static final int BLOCK_SIZE = 25; // this is just used for rendering
public static final int GAME_SIZE = 6; // this is the game width and height
public static final int NUM_THREADS = 4; // can only handle 8 threads max
public static final int SNAKES_PER_THREAD = 4; // number of snakes per thread, total snakes = 4 * 4 = 16
public static final int rollout = 5; // number of steps to take before training the network on the experiences its collected
public static final int actions = 4; // number of actions available (up, down, left, right) basically the action space
public static final double gamma = 0.9; // discount rate, discounts rewards to balance the value between rewards now and in the future
public static final double value_loss_coeff = 0.5; // scales the critic head loss
public static final double entropy_loss_coeff = 0.01; // entropy (measures uncertainty in the network) coefficient, used for exploration
public static final double lr = 0.001; // learning rate
public static final double l2 = 0.000001; // l2 coefficient, this determines the rate of weight decay
public static final long seed = 0xC0FFEE; // the seed used for random number generation
public static final int[] observation_space = new int[] {GAME_SIZE, GAME_SIZE, 3}; // width, height, channels
public static final String DIR = "C:\\Users\\saraa\\Desktop\\snakes\\"; // directory to save files to
public static final int WIDTH = GAME_SIZE*BLOCK_SIZE, HEIGHT = WIDTH; // also used for rendering
public static final boolean convnet = false; // true = convnet, false = simplenet
public static final int saveEvery = 500; // save every n training steps
public static final int printEvery = 1000; // print summaries every n game steps
public static final int trainingSteps = 1000000; // train for this many steps
public static final Random rand = new Random(seed); // random number generator

	public static void main(String[] args) {
//		Nd4j.getRandom().setSeed(seed);
		Nd4j.getMemoryManager().setAutoGcWindow(10000);
		TrainingThread trainer = new TrainingThread();
		trainer.start();
	}
}


