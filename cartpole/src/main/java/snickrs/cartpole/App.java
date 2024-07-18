package snickrs.cartpole;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import processing.core.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.XavierInitScheme;
import processing.core.*;
import java.util.Collections;
import java.text.DecimalFormat;
import java.util.Random;

public class App extends PApplet {
	PPO model = new PPO(0.001, 0.2, 0.99, 0.95, 8);
	CartpoleEnvironment env = new CartpoleEnvironment();
	INDArray state = env.getState();
    public static void main( String[] args ) {
    	String[] processingArgs = {"App"};
		App mySketch = new App();
		PApplet.runSketch(processingArgs, mySketch);
    }
    public void settings() {
    	size(600, 400);
    }
    public void setup() {
    	frameRate(10000);
    	textAlign(CENTER, CENTER);
    	strokeCap(SQUARE);
    }
    public void draw() {
    	background(255);
    	
    	INDArray probs = model.output(state);
		int action = model.sample(probs);
		env.step(action);
		env.render(this);
		fill(0);
		textSize(24);
		text("STEPS: " + env.steps, 100, 100);
		text("Current Reward: " + env.cumreward, 200, 150);
		model.game.saveData(state, probs, action, env.reward, env.terminated);
		state = env.getState();
		
		if(env.steps % 320 == 0) {
			model.game.stateLs.add(state);
			model.learn();
		}
    	if(env.terminated) {
			env.reset();
			state = env.getState();
		}
		
		if(env.win) noLoop();
    }
    public void train() {
    	for(int i = 0; i < 100000; i++) {
    		if(env.terminated) {
    			env.reset();
    			state = env.getState();
    		}
    		INDArray probs = model.output(state);
    		int action = model.sample(probs);
    		env.step(action);
    		model.game.saveData(state, probs, action, env.reward, env.terminated);
    		state = env.getState();
    		
    		if(env.steps % 320 == 0) {
    			model.game.stateLs.add(state);
    			model.learn();
    		}
    		if(env.win) break;
    	}
    }
    void graph(ArrayList<PVector> data) {
    	  DecimalFormat format = new DecimalFormat("0.#");
    	  float ticks = 10;
    	  //the max value is actually 500 but we wanna make the graph look sexy
    	  float maxVal = 525;
    	  PVector scale = new PVector(data.get(data.size()-1).x/ticks, maxVal/ticks);
    	  PVector dist = new PVector((width-48)/data.size(), (height-48)/maxVal);

    	  //graph axes
    	  fill(0);
    	  stroke(0);
    	  strokeWeight(1);
    	  textSize(16);
    	  text("Timesteps", width/2, height-16);
    	  line(48, height-48, width, height-48);
    	  line(48, 0, 48, height-48);
    	  pushMatrix();
    	  translate(8, height/2);
    	  rotate(3*PI/2);
    	  text("Reward", 0, 0);
    	  popMatrix();

    	  //graph ticks
    	  textSize(12);
    	  for (int i = 0; i < ticks; i++) {
    	    text(format.format(i*scale.x), 48+(width-32)/10*i, height-30);
    	    line(48+(width-32)/10*i, height-56, 48+(width-32)/10*i, height-40);
    	  }
    	  for (int i = 0; i < ticks; i++) {
    	    text(format.format(i*scale.y), 32-6, (height-48)-2-(height-32)/10*i);
    	    line(40, (height-48)-(height-32)/10*i, 56, (height-48)-(height-32)/10*i);
    	  }
    	  //draw graph
    	  stroke(255, 0, 0);
    	  for (int i = 0; i < data.size()-1; i++) {
    	    line(48+i*dist.x, (height-48)-2-dist.y*data.get(i).y, 48+(i+1)*dist.x, (height-48)-2-dist.y*data.get(i+1).y);
    	  }
    	}
}
