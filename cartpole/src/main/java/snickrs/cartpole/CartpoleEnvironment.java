package snickrs.cartpole;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import processing.core.PApplet;
import processing.core.PShape;
import processing.core.PVector;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
class CartpoleEnvironment {
	  Random random = new Random(0);
	  double screen_width = 600;
	  double screen_height = 400;
	  int steps = 0;
	  boolean win = false;
	  double gravity = 9.8f;
	  double masscart = 1.0f;
	  double masspole = 0.1f;
	  double total_mass = masspole+masscart;
	  double length = 0.5f; //half the poles length
	  double polemass_length = masspole*length;
	  double force_mag = 10.0f;
	  double tau = 0.02f;
	  // Angle at which to fail the episode
	  double theta_threshold_radians = 12 * 2 * Math.PI / 360;
	  double x_threshold = 2.4f;
	  double[] high = new double[] {x_threshold * 2, Double.MAX_VALUE, theta_threshold_radians * 2, Double.MAX_VALUE};
	  double[] action_space = new double[] {0, 1};
	  double[][] observation_space = new double[][] {{-4.8f, 4.8f}, {-Double.MAX_VALUE, Double.MAX_VALUE}, {-0.418879f, 0.418879f}, {-Double.MAX_VALUE, Double.MAX_VALUE}};
	  double[] state;
	  boolean terminated = false;
	  double reward = 0;
	  double cumreward = 0;
	  double world_width = x_threshold * 2;
	  double scale = screen_width / world_width;
	  double polewidth = 10.0f;
	  double polelen = scale * (2 * length);
	  double cartwidth = 50.0f;
	  double cartheight = 30.0f;
	  int counter = 0;
	  ArrayList<PVector> points = new ArrayList<PVector>();

	  CartpoleEnvironment() {
		  state = new double[] {-0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble()};
	  }

	  void step(int action) {
	    steps++;
	    counter++;
	    double x = state[0];
	    double x_dot = state[1];
	    double theta = state[2];
	    double theta_dot = state[3];
	    double force = action == 1 ? force_mag : -force_mag;
	    double costheta = Math.cos(theta);
	    double sintheta = Math.sin(theta);
	    double temp = (force + polemass_length * Math.pow(theta_dot, 2) * sintheta) / total_mass;
	    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0f / 3.0f - masspole * Math.pow(costheta, 2) / total_mass));
	    double xacc = temp - polemass_length * thetaacc * costheta / total_mass;
	    x = x + tau * x_dot;
	    x_dot = x_dot + tau * xacc;
	    theta = theta + tau * theta_dot;
	    theta_dot = theta_dot + tau * thetaacc;
	    state = new double[] {x, x_dot, theta, theta_dot};
	    terminated = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
	    if (cumreward >= 500) {
	      System.out.println("WIN IN " + Integer.toString(steps) + " STEPS");
	      points.add(new PVector((float) steps, (float) cumreward));
	      win = true;
	    }
	    if (terminated) {
	      reward = -1;
	      points.add(new PVector((float) steps, (float) cumreward));
	      cumreward = 0;
	    } else {
	      reward = 1;
	      cumreward += 1;
	    }
	  }
	  INDArray getState() {
		  return Nd4j.create(state, new int[] {1, state.length});
	  }
	  void reset() {
		  state = new double[] {-0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble(), -0.05d + (0.05d - (-0.05d)) * random.nextDouble()};
		  terminated = false;
	  }
	  void render(PApplet p) {
		    double[] x = state.clone();
		    float l = (float) (-cartwidth/2);
		    float r = (float) (cartwidth/2);
		    float t = (float) (cartheight/2);
		    float b = (float) (-cartheight/2);
		    float axleoffset = (float) (cartheight / 4.0f);
		    float cartx = (float) (x[0] * scale + screen_width / 2.0);  // MIDDLE OF CART
		    float carty = 100; // TOP OF CART
		    float[][] cart_coords = new float[][] {{l, b}, {l, t}, {r, t}, {r, b}};
		    for (int i = 0; i < cart_coords.length; i++) {
		      cart_coords[i][0] += cartx;
		      cart_coords[i][1] += carty;
		    }
		    p.stroke(0);
		    p.strokeWeight(1);
		    p.fill(0);
		    p.rect(cart_coords[1][0], p.height-cart_coords[1][1], (float) cartwidth, (float) cartheight);
		    l = (float) (-polewidth/2);
		    r = (float) (polewidth/2);
		    t = (float) (polelen - polewidth/2);
		    b = (float) (-polewidth/2);
		    float[][] pole_coords = new float[4][2];
		    float[][] arr = new float[][] {{l, b}, {l, t}, {r, t}, {r, b}};
		    for (int i = 0; i < arr.length; i++) {
		      float[] c = arr[i];
		      PVector coord = new PVector(c[0], c[1]);
		      coord.rotate((float) -x[2]);
		      coord.x += cartx;
		      coord.y += carty + axleoffset;
		      pole_coords[i] = new float[] {coord.x, coord.y};
		    }
		    p.stroke(202, 152, 101);
		    PShape shape = p.createShape();
		    shape.beginShape();
		    shape.fill(202, 152, 101);
		    shape.vertex(pole_coords[0][0], p.height-pole_coords[0][1]);
		    shape.vertex(pole_coords[1][0], p.height-pole_coords[1][1]);
		    shape.vertex(pole_coords[2][0], p.height-pole_coords[2][1]);
		    shape.vertex(pole_coords[3][0], p.height-pole_coords[3][1]);
		    shape.endShape();
		    p.shape(shape);
		    p.fill(129, 132, 203);
		    p.stroke(0);
		    //buggy for some reason
		    p.ellipse(cartx, p.height-(carty+axleoffset), (float) polewidth, (float) polewidth);
		    p.stroke(0);
		    p.line(0, p.height-carty, (float) screen_width, p.height-carty);
		  }
	}
