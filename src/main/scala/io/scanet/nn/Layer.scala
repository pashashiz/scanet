package io.scanet.nn

import breeze.linalg.DenseMatrix
import io.scanet.core.Product
import simulacrum.{op, typeclass}


// Layer can be converted to diff function
// SGD.optimize((Dense(1, 2) <+> Dense(1, 2)).toDFBuilder[A])
// or with ToDFBuilder[A]Opt
// SGD.optimize(Dense(1, 2) <+> Dense(1, 2))


@typeclass trait Layer[L] extends Product[L] {

  /**
    * Forward propagation
    *
    * @param layer Layer
    * @param theta [OUTxIN] matrix where each row has a vector of coefficients for a neuron (OUT neurons, IN coefficients)
    * @param input [input: MxIN] matrix where each row is a an item from training set containing a vector of features (M rows, IN features)
    * @return [MxOUT] matrix where each row will contain all activations for an item from training set
    */
  def forward(layer: L, theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double]

  /**
    * Backward propagation
    *
    * @param layer Layer
    * @param theta [OUTxIN] matrix where each row has a vector of coefficients for a neuron (OUT neurons, IN coefficients)
    * @param input [input: MxIN] matrix where each row is a an item from training set containing a vector of features (M rows, IN features)
    * @param error [output: MxOUT] matrix where each row is a an item from training set containing a vector of activation errors for each neuron (M rows, OUT neurons)
    * @return [OUTxIN] coefficient gradients
    */
  def backprop(layer: L, theta: List[DenseMatrix[Double]], input: DenseMatrix[Double], error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]])

  def power(layer: L): Int = 1

}
