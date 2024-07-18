package snickrs.snakeai;

import processing.core.PApplet;
import java.time.format.DateTimeFormatter;

import org.deeplearning4j.nn.graph.ComputationGraph;

import java.time.LocalDateTime;    
public class SnakeThread extends Thread {
  public Snake[] snakes = new Snake[SnakeAI.SNAKES_PER_THREAD];
  public int ind = 0;
  
  public SnakeThread(ComputationGraph ai, int index) {
	  for(int i = 0; i < snakes.length; i++) {
		  snakes[i] = new Snake(ai);
	  }
	  ind = index;
  }
  public SnakeThread(PApplet p, ComputationGraph ai, int index) {
	  for(int i = 0; i < snakes.length; i++) {
		  snakes[i] = new Snake(p, ai);
	  }
	  ind = index;
  }
  public void run() {
    while (!Thread.currentThread().isInterrupted()) {
    	for(int i = 0; i < snakes.length; i++) {
    	if(snakes[i].updateSnake) {
    		snakes[i].gamestep();
//    		PApplet.println(snake.game.actionLs);
        	if(snakes[i].steps % SnakeAI.printEvery == 0) {
        		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");  
        		LocalDateTime now = LocalDateTime.now();  
        		System.out.println(dtf.format(now) + " snake " + (ind*SnakeAI.NUM_THREADS+i) + ", steps: " + snakes[i].steps + ", games played: " + snakes[i].games + ", max score: " + snakes[i].maxScore);
        	}
    	}
    }
    }
  }
}

