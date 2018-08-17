package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.func.DiffFunction
import DiffFunction.ops._

case class Dense[A](units: Int, activation: A)

trait DenseLayerInst {

  implicit def DenseLayer[A: DiffFunction]: Layer[Dense[A]] = new Layer[Dense[A]] {

    /**
      * Forward propagation
      *
      * @param layer Layer
      * @param theta [OUTxIN] matrix where each row has a vector of coefficients for a neuron (OUT neurons, IN coefficients)
      * @param input [input: MxIN] matrix where each row is a an item from training set containing a vector of features (M rows, IN features)
      * @return [MxOUT] matrix where each row will contain all activations for an item from training set
      */
    override def forward(layer: Dense[A], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double] = {
      layer.activation.apply1(input * theta.head.t)
    }

    /**
      * Backward propagation
      *
      * @param layer Layer
      * @param theta [OUTxIN] matrix where each row has a vector of coefficients for a neuron (OUT neurons, IN coefficients)
      * @param input [input: MxIN] matrix where each row is a an item from training set containing a vector of features (M rows, IN features)
      * @param error [output: MxOUT] matrix where each row is a an item from training set containing a vector of activation errors for each neuron (M rows, OUT neurons)
      * @return [OUTxIN] coefficient gradients
      */
    override def backprop(layer: Dense[A], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double], error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]]) = {
      // NOTE: for output layer
      val grad = layer.activation.gradient1(input * theta.head.t) // MxOUT
      val delta: DenseMatrix[Double] = grad *:* error // MxOUT
      (delta, List(delta.t * input))
    }
  }
}
