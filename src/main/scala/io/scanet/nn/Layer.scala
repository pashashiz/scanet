package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.core.Product
import simulacrum.typeclass


@typeclass trait Layer[L] extends Product[L] {

  /**
    * Forward propagation
    *
    * @param layer Layer
    * @param theta [OUT x (IN + 1)] matrix where each row has a vector of coefficients
    *              for a neuron (OUT neurons, IN coefficients, 1 bias)
    * @param input [input: M x IN] matrix where each row is a an item from training set
    *              containing a vector of features (M rows, IN features)
    * @return [M x OUT] matrix where each row will contain all activations for an item from training set
    */
  def forward(layer: L, theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double]

  /**
    * Backward propagation
    *
    * @param layer Layer
    * @param theta [OUT x (IN + 1)] matrix where each row has a vector of coefficients
    *              for a neuron (OUT neurons, IN coefficients, 1 bias)
    * @param input [input: M x IN] matrix where each row is a an item from training set
    *              containing a vector of features (M rows, IN features)
    * @param error [output: M x OUT] matrix where each row is a an item from training set containing
    *              a vector of activation errors for each neuron (M rows, OUT neurons)
    * @return [OUT x IN] coefficient gradients
    */
  def backprop(layer: L, theta: List[DenseMatrix[Double]], input: DenseMatrix[Double], error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]])

  def power(layer: L): Int = 1

  def shape(layer: L): List[Shape]

  def pack(layer: L, vars: List[DenseMatrix[Double]]): DenseVector[Double] = {
    vars.foldLeft(DenseVector[Double]())(
      (acc, layer) => DenseVector.vertcat(acc, layer.t.toDenseVector))
  }

  def unpack(layer: L, inputCols: Int)(vars: DenseVector[Double]): List[DenseMatrix[Double]] = {
    def go(shape: List[Shape], skip: Int, acc: List[DenseMatrix[Double]]): List[DenseMatrix[Double]] = shape match {
      case Nil => acc
      case first::second::tail =>
        val firstUnits = if (second.bias) first.units + 1 else first.units
        val end = skip + firstUnits * second.units
        val layer = vars(skip until end).toDenseMatrix.reshape(firstUnits, second.units).t
        go(if (tail != Nil) second::tail else Nil, end, layer :: acc)
    }
    val fullShape = Shape(inputCols, bias = false) :: shape(layer)
    go(fullShape, 0, List()).reverse
  }
}

case class Shape(units: Int, bias: Boolean)
