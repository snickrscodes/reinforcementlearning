package snickrs.snakeai;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;

public class TrainingThread extends Thread {
	  public SnakeThread[] snakes = new SnakeThread[SnakeAI.NUM_THREADS];
	  public ActorCriticNetwork ai = new ActorCriticNetwork();
	  public int trainingSteps = 0;
	  public TrainingThread() {
//		  load();
		  for(int i = 0; i < snakes.length; i++) {
			  snakes[i] = new SnakeThread(ai.agent, i);
			  snakes[i].start();
		  }
	  }
	  public boolean allSnakesDead() {
			for(int i = 0; i < snakes.length; i++) {
				for(int j = 0; j < snakes[i].snakes.length; j++) {
    				if(snakes[i].snakes[j].updateSnake) return false;
    			}
			}
			return true;
		}
	  public void load() {
		  try {
			ai.agent = ComputationGraph.load(new File(SnakeAI.DIR+"snake.dl4j"), true);
		} catch (IOException e) {
			System.err.println("couldnt load network");
		}
	  }
	  public void save() {
          	try {
				ai.agent.save(new File(SnakeAI.DIR+"petsnake.dl4j"), true);
				System.out.println("model saved");
			} catch (IOException e) {
				System.err.println("couldnt save network");
			}
	  }
	  public void run() {
	    while (!Thread.currentThread().isInterrupted()) {
	    	if(allSnakesDead()) {
	    		Game[] games = new Game[snakes.length];
	    		for(int i = 0; i < snakes.length; i++) {
//	    			games[i] = snakes[i].snake.game;
	    			for(int j = 0; j < snakes[i].snakes.length; j++) {
	    				games[i] = snakes[i].snakes[j].game.clone();
	    			}
//	    			ai.fit(snakes[i].snake.game);
	    		}
	    		ai.fit(games);
	        	for(int i = 0; i < snakes.length; i++) {
	        		for(int j = 0; j < snakes[i].snakes.length; j++) {
	        			snakes[i].snakes[j].game.clear();
	        		}
	        	}
	        	System.gc();
	    		trainingSteps++;
	    		if(trainingSteps % SnakeAI.saveEvery == 0) save();
	    		for(int i = 0; i < snakes.length; i++) {
	    			for(int j = 0; j < snakes[i].snakes.length; j++) {
	    				snakes[i].snakes[j].policy = ai.agent.clone();
	    			}
//	    			snakes[i].snake.policy = ai.agent.clone();
	    		}
	    		for(int i = 0; i < snakes.length; i++) {
	    			for(int j = 0; j < snakes[i].snakes.length; j++) {
	    				if(!snakes[i].snakes[j].win) snakes[i].snakes[j].updateSnake = true;
	    			}
//	    			if(!snakes[i].snake.win) snakes[i].snake.updateSnake = true;
	    		}
	    		System.gc();
	    	}
	    }
	  }
}

