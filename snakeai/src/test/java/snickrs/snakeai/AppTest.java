package snickrs.snakeai;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

public class AppTest {
    public static void main(String[] args) {
    	DefaultRandom rand = new DefaultRandom(0);
    	System.out.println(rand.nextDouble(new int [] {1, 1}));
    }
}
