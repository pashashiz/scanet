package io.scanet.nn

import breeze.linalg.DenseMatrix
import io.scanet.func.DiffFunction
import DiffFunction.ops._

case class Dense[A](units: Int, activation: A, bias: Boolean = true)

trait DenseLayerInst {

  implicit def DenseLayer[A: DiffFunction]: Layer[Dense[A]] = new Layer[Dense[A]] {

    override def forward(layer: Dense[A], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double] = {
      val inputB = if (layer.bias) DenseMatrix.horzcat(DenseMatrix.ones[Double](input.rows, 1), input) else input
      val f = inputB * theta.head.t
      layer.activation.apply1(f)
    }

    override def backprop(layer: Dense[A], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double], error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]]) = {
      val inputB = if (layer.bias) DenseMatrix.horzcat(DenseMatrix.ones[Double](input.rows, 1), input) else input
      val grad = layer.activation.gradient1(inputB * theta.head.t) // M x OUT
      val delta: DenseMatrix[Double] = grad *:* error // M x OUT
      (delta, List(delta.t * inputB))
    }

    override def shape(layer: Dense[A]): List[Shape] = List(Shape(layer.units, layer.bias))
  }
}
