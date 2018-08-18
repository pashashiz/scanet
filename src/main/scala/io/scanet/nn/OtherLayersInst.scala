package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.nn.Layer.ops._


trait OtherLayersInst {

  implicit def TupleLayer[A: Layer, B: Layer]: Layer[(A, B)] = new Layer[(A, B)] {

    /**
      * Forward propagation
      *
      * @param layer Layer
      * @param theta [OUTxIN] matrix where each row has a vector of coefficients for a neuron (OUT neurons, IN coefficients)
      * @param input [input: MxIN] matrix where each row is a an item from training set containing a vector of features (M rows, IN features)
      * @return [MxOUT] matrix where each row will contain all activations for an item from training set
      */
    override def forward(layer: (A, B), theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double] = {
      val (left, right) = layer
      val leftTheta = theta.take(left.power)
      val rightTheta = theta.slice(left.power, left.power + right.power)
      right.forward(rightTheta, left.forward(leftTheta, input))
    }

    /**
      * Backward propagation
      *
      * @param layer Layer
      * @param theta [OUT x IN] matrix where each row has a vector of coefficients
      *              for a neuron (OUT neurons, IN coefficients) for every layer
      * @param input [input: M x IN] matrix where each row is a an item from training set
      *              containing a vector of features (M rows, IN features)
      * @param error [output: M x OUT] matrix where each row is a an item from training set
      *              containing a vector of activation errors for each neuron (M rows, OUT neurons)
      * @return [OUT x IN] coefficient gradients for every layer
      */
    override def backprop(layer: (A, B), theta: List[DenseMatrix[Double]], input: DenseMatrix[Double],
                          error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]]) = {
      val (left, right) = layer
      val leftTheta = theta.take(left.power) // OUT_l x IN
      val rightInput = left.forward(leftTheta, input) // M x OUT_r
      val rightTheta = theta.slice(left.power, left.power + right.power) // OUT_r x OUT_l
      val (rightDelta, rightGrad) = right.backprop(rightTheta, rightInput, error) // M x OUT_r, OUT_r x OUT_l
      val leftError = rightDelta * rightTheta.head // M x OUT_r * OUT_r x OUT_l = M x OUT_l
      val (leftDelta, leftGrad) = left.backprop(leftTheta, input, leftError) // OUT_l x IN
      (leftDelta, leftGrad ++ rightGrad)
    }

    override def power(layer: (A, B)): Int = layer match {
      case (left, right) => left.power + right.power
    }

    override def shape(layer: (A, B)): List[Int] = layer match {
      case (left, right) => left.shape ++ right.shape
    }
  }
}


