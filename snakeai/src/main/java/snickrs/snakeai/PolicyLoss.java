package snickrs.snakeai;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

public class PolicyLoss implements ILossFunction {

	 /**
	 * 
	 */
	private static final long serialVersionUID = -2889153679526041173L;

	@Override
     public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
         return 0;
     }

     @Override
     public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
         return null;
     }
     
     public INDArray entropy(INDArray logits) {
    	 INDArray a0 = logits.sub(logits.max(true, -1));
    	 INDArray ea0 = Transforms.exp(a0, true);
    	 INDArray z0 = ea0.sum(true, -1);
    	 INDArray p0 = ea0.div(z0);
    	 INDArray r0 = Transforms.log(z0, true).sub(a0);
    	 INDArray t0 = p0.mul(r0);
    	 return t0.sum(true, -1);
     }
     
     @Override
     public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
         INDArray output = activationFn.getActivation(preOutput.dup(), true).add(1e-5);
         INDArray logOut = Transforms.log(output, true);
         INDArray entropy = entropy(output).mul(SnakeAI.entropy_loss_coeff).neg();
         INDArray loss = logOut.mul(labels).add(entropy);
         INDArray gradient = activationFn.backprop(preOutput, loss).getFirst();
         return gradient;
     }

     @Override
     public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
         return null;
     }

     @Override
     public String name() {
         return null;
     }
}
