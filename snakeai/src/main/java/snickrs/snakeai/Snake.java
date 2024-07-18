package snickrs.snakeai;

import processing.core.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Snake {
		  public ArrayList<PVector> body;
		  public PVector vel;
		  public boolean render = true;
		  public int steps = 0;
		  public int score = 0;
		  public int frames = 0;
		  public PVector apple = new PVector(0, 0);
		  public PApplet p;
		  public boolean win = false;
		  public boolean updateSnake = true;
		  public int maxScore = 0;
		  public int games = 0;
		  public INDArray state;
		  public int counter = 0;
		  public Game game;
		  public boolean terminated = false;
//		  public ActorCriticNetwork ai;
		  public ComputationGraph policy;
		  public Snake (PApplet parent, ComputationGraph ai) {
		    body = new ArrayList<PVector>();
		    body.add(new PVector(SnakeAI.GAME_SIZE/2, SnakeAI.GAME_SIZE/2));
		    vel = new PVector(0, 0);
		    food();
		    p = parent;
		    game = new Game();
//		    this.ai = ai;
		    policy = ai.clone();
		    state = getInput();
		  }
		  public Snake (ComputationGraph ai) {
			    body = new ArrayList<PVector>();
			    body.add(new PVector(SnakeAI.GAME_SIZE/2, SnakeAI.GAME_SIZE/2));
			    vel = new PVector(0, 0);
			    food();
			    game = new Game();
//			    this.ai = ai;
			    policy = ai.clone();
			    render = false;
			    state = getInput();
			  }
		  void bounds() {
			  p.stroke(255);
			  p.line(0, 0, 0, SnakeAI.HEIGHT);
			  p.line(SnakeAI.WIDTH, 0, SnakeAI.WIDTH, SnakeAI.HEIGHT);
			  p.line(0, 0, SnakeAI.WIDTH, 0);
			  p.line(0, SnakeAI.HEIGHT, SnakeAI.WIDTH, SnakeAI.HEIGHT);
			  p.stroke(0);
			}
		  void food () {
			  ArrayList<PVector> spaces = new ArrayList<PVector>();
			  for (int i = 0; i < SnakeAI.GAME_SIZE; i++) {
			    for (int j = 0; j < SnakeAI.GAME_SIZE; j++) {
			      PVector vec = new PVector(i, j);
			      if (!collides(vec)) spaces.add(vec);
			    }
			  }
			  if(spaces.size() == 0) {
				  updateSnake = false;
				  win = true;
				  try {
					policy.save(new File(SnakeAI.DIR+"snake6x6.dl4j"), true);
				} catch (IOException e) {
					System.err.println("didnt save the fucking file, but the snake won");
				}
				  System.out.println("win after " + Integer.toString(steps) + " steps, and " + Integer.toString(games) + " games.");
				  System.exit(0);
				  return;
			  }
			  apple = spaces.get((int) SnakeAI.rand.nextInt(0, spaces.size())).copy();
			}
		  void update () {
		    PVector head = body.get(body.size() - 1).copy();
		    body.remove(0);
		    head.add(vel);
		    body.add(head);
		  }
		  void grow () {
		    PVector head = body.get(body.size() - 1).copy();
		    body.add(head);
		    score++;
		  }
		  void move(int act) {
			if(act == 0) vel = new PVector(0, -1);
			else if(act == 1) vel = new PVector(0, 1);
			else if(act == 2) vel = new PVector(-1, 0);
			else if(act == 3)vel = new PVector(1, 0);
		  }
		  boolean endGame() {
		    float x = body.get(body.size() - 1).x;
		    float y = body.get(body.size() - 1).y;
		    if (x > SnakeAI.GAME_SIZE-1 || x < 0 || y > SnakeAI.GAME_SIZE-1 || y < 0) {
		      return true;
		    }
		    for (int i = 0; i < body.size()-1; i++) {
		      PVector part = body.get(i);
		      if (part.x == x && part.y == y) {
		        return true;
		      }
		    }
		    return false;
		  }
		  boolean oob(PVector p) {
			  return p.x > SnakeAI.GAME_SIZE-1 || p.x < 0 || p.y > SnakeAI.GAME_SIZE-1 || p.y < 0;
		  }
		  boolean collides(PVector point) {
		    for (int i = 0; i < body.size(); i++) {
		      PVector part = body.get(i);
		      if (part.x == point.x && part.y == point.y) {
		        return true;
		      }
		    }
		    return false;
		  }
		  boolean eat(PVector pos) {
		    float x = body.get(body.size() - 1).x;
		    float y = body.get(body.size() - 1).y;
		    if (x == pos.x && y == pos.y) {
		      grow();
		      return true;
		    }
		    return false;
		  }
		  float conv(boolean b) {
		    return b ? 1 : 0;
		  }
		  INDArray getInput() {
			  if (SnakeAI.convnet) {
			  float[][][] input = new float[SnakeAI.observation_space[2]][SnakeAI.observation_space[1]][SnakeAI.observation_space[0]];
			  for(int i = 0; i < input.length; i++) {
				  for(int j = 0; j < input[i].length; j++) {
					  for(int k = 0; k < input[i][j].length; k++) {
					  input[i][j][k] = 0;
					  }
				  }
			  	}
//			  for(int i = 0; i < SnakeAI.observation_space[1]; i++) {
//				  input[0][i][0] = 1;
//				  input[0][i][SnakeAI.GAME_SIZE+1] = 1;
//			  }
//			  for(int i = 0; i < SnakeAI.observation_space[0]; i++) {
//				  input[0][0][i] = 1;
//				  input[0][SnakeAI.GAME_SIZE+1][i] = 1;
//			  }
//			  PVector head = body.get(body.size()-1).copy();
//			  input[0][(int)apple.y+1][(int)apple.x+1] = -1;
//			  input[1][(int)head.y+1][(int)head.x+1] = 1;
//			  for(int i = 0; i < body.size()-1; i++) {
//				  input[0][(int)body.get(i).y][(int)body.get(i).x] = 1;
//			  	}
			  PVector head = body.get(body.size()-1).copy();
			  input[0][(int)apple.y][(int)apple.x] = 1;
			  if(!oob(head)) input[1][(int)head.y][(int)head.x] = 1;
			  for(int i = 0; i < body.size()-1; i++) {
				  input[2][(int)body.get(i).y][(int)body.get(i).x] = 1;
			  	}
			  return Nd4j.create(new float[][][][] {input});
			  } else {
				    PVector head = body.get(body.size()-1).copy();
				    PVector point_l = new PVector(head.x - 1, head.y);
				    PVector point_r = new PVector(head.x + 1, head.y);
				    PVector point_u = new PVector(head.x, head.y - 1);
				    PVector point_d = new PVector(head.x, head.y + 1);
				    boolean dir_l = vel.x == -1;
				    boolean dir_r = vel.x == 1;
				    boolean dir_u = vel.y == -1;
				    boolean dir_d = vel.y == 1;
				    boolean[] input = new boolean[] {
				      (dir_r && collides(point_r)) || (dir_l && collides(point_l)) || (dir_u && collides(point_u)) || (dir_d && collides(point_d)),
				      (dir_u && collides(point_r))|| (dir_d && collides(point_l))|| (dir_l && collides(point_u))|| (dir_r && collides(point_d)),
				      (dir_d && collides(point_r))|| (dir_u && collides(point_l))|| (dir_r && collides(point_u))|| (dir_l && collides(point_d)),
				      dir_l,
				      dir_r,
				      dir_u,
				      dir_d,
				      apple.x < head.x,
				      apple.x > head.x,
				      apple.y < head.y,
				      apple.y > head.y
				    };
				    return Nd4j.create(new boolean[][] {input});
			  	}
			  }
			int argmax(INDArray arr) {
				double[] a = arr.data().asDouble();
				  double ref = -1000;
				  int arg = -1;
				  for (int i = 0; i < a.length; i++) {
				    if (a[i] > ref) {
				      ref = a[i];
				      arg = i;
				    }
				  }
				  return arg;
				}
			int sample(INDArray arr) {
				  double[] probs = arr.data().asDouble();
				  double x = SnakeAI.rand.nextDouble();
				  for (int i = 0; i < probs.length; ++i) {
				    x -= probs[i];
				    if (x <= 0) {
				      return i;
				    }
				  }
				  return probs.length-1;
				}
			void info() {
			  p.fill(255);
			  p.textSize(24);
			  p.textAlign(PConstants.CORNER);
			  p.text("Score: " + score, 0, SnakeAI.HEIGHT+24);
			  p.text("Max Score: " + maxScore, 0, SnakeAI.HEIGHT+48);
			  p.text("Games: " + games, 0, SnakeAI.HEIGHT+72);
		  }
		  public void gamestep() {
			if(!updateSnake) return;
			steps++;
			counter++;
			if(render) {
			bounds();
			info();
			}
			if(terminated) {
//		    	game.stateLs.add(state);
//		    	ai.learn(index);
				reset();
				state = getInput();
//				updateSnake = false;
//				return;
			}
			frames++;
			INDArray probs = policy.output(new INDArray[] {state})[0];
		    int ac = sample(probs);
		    move(ac);
		    if(render) show();
		    update();
		    double reward = getReward();
		    maxScore = Math.max(score, maxScore);
		    game.saveData(state, probs, ac, reward, terminated);
		    state = getInput();
		    if(counter >= SnakeAI.rollout) {
		    	counter = 0;
		    	game.stateLs.add(state);
		    	updateSnake = false;
		    }
		  }
		  double getReward() {
			    if (endGame() || frames > (SnakeAI.GAME_SIZE)*(SnakeAI.GAME_SIZE)) {
			    	terminated = true;
			    	return -5;
			    }
			    if (eat(apple)) {
			    	food();
			    	return 1;
			    }
			    return -0.14; // approximately -5/36 = reward for losing / number of steps without food to lose
		  }
		  void reset() {
		    body.clear();
		    body.add(new PVector(SnakeAI.GAME_SIZE/2, SnakeAI.GAME_SIZE/2));
		    vel = new PVector(0, 0);
		    terminated = false;
		    score = 0;
		    frames = 0;
		    games++;
		    food();
		  }
		  void show() {
		    PVector head = body.get(body.size()-1).copy();
		    p.noStroke();
		    p.fill(0, 255, 0);
		    float len = SnakeAI.BLOCK_SIZE/17f;
		    p.rect(head.x*SnakeAI.BLOCK_SIZE+len, head.y*SnakeAI.BLOCK_SIZE+len, SnakeAI.BLOCK_SIZE-len*2, SnakeAI.BLOCK_SIZE-len*2);
		    for (int i = 0; i < body.size()-1; i++) {
		      p.rect(body.get(i).x*SnakeAI.BLOCK_SIZE+len, body.get(i).y*SnakeAI.BLOCK_SIZE+len, SnakeAI.BLOCK_SIZE-len*2, SnakeAI.BLOCK_SIZE-len*2);
		      p.rect((body.get(i).x*SnakeAI.BLOCK_SIZE+body.get(i+1).x*SnakeAI.BLOCK_SIZE)/2+len, (body.get(i).y*SnakeAI.BLOCK_SIZE+body.get(i+1).y*SnakeAI.BLOCK_SIZE)/2+len, SnakeAI.BLOCK_SIZE-len*2, SnakeAI.BLOCK_SIZE-len*2);
		    }
			  p.fill(255, 0, 0);
			  p.rect(apple.x*SnakeAI.BLOCK_SIZE+len, apple.y*SnakeAI.BLOCK_SIZE+len, SnakeAI.BLOCK_SIZE-len*2, SnakeAI.BLOCK_SIZE-len*2);
		  }

}

