package snickrs.snakeai;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.common.primitives.Pair;

public class ValueLoss implements ILossFunction {

	 /**
	 * 
	 */
	private static final long serialVersionUID = 1651010965519539613L;

	@Override
     public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
         return 0;
     }

     @Override
     public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
         return null;
     }
     
     @Override
     public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
         INDArray output = activationFn.getActivation(preOutput.dup(), true);
         INDArray loss = output.subi(labels).muli(2*SnakeAI.value_loss_coeff).divi(labels.size(1));
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
