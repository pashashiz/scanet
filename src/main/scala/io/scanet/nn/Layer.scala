package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.core.Product
import simulacrum.typeclass


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

  def shape(layer: L): List[Int]

  def pack(layer: L, vars: List[DenseMatrix[Double]]): DenseVector[Double] = {
    vars.foldLeft(DenseVector[Double]())(
      (acc, layer) => DenseVector.vertcat(acc, layer.t.toDenseVector))
  }

  def unpack(layer: L, inputCols: Int)(vars: DenseVector[Double]): List[DenseMatrix[Double]] = {
    def go(shape: List[Int], skip: Int, acc: List[DenseMatrix[Double]]): List[DenseMatrix[Double]] = shape match {
      case Nil => acc
      case first::second::tail =>
        val end = skip + first * second
        val layer = vars(skip until end).toDenseMatrix.reshape(first, second).t
        go(if (tail != Nil) second::tail else Nil, end, layer :: acc)
    }
    val fullShape = inputCols :: shape(layer)
    go(fullShape, 0, List()).reverse
  }

}
