package io.scanet.nn

import breeze.linalg._
import io.scanet.core.DiffFunction
import io.scanet.core.DiffFunction.ops._
import io.scanet.core.func.Zero

case class Dense[A: DiffFunction, KR: DiffFunction, BR: DiffFunction](
                     units: Int,
                     activation: A,
                     bias: Boolean = true,
                     kernelReg: KR = Zero(),
                     biasReg: BR = Zero())

trait DenseLayerInst {

  implicit def DenseLayer[A: DiffFunction, KR: DiffFunction, BR: DiffFunction]: Layer[Dense[A, KR, BR]] = new Layer[Dense[A, KR, BR]] {

    override def shape(layer: Dense[A, KR, BR]): List[Shape] = List(Shape(layer.units, layer.bias))

    override def forward(layer: Dense[A, KR, BR], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double] = {
      val inputAndBias = if (layer.bias) DenseMatrix.horzcat(DenseMatrix.ones[Double](input.rows, 1), input) else input
      val f = inputAndBias * theta.head.t
      layer.activation.apply1(f)
    }

    override def forwardPenalty(layer: Dense[A, KR, BR], theta: List[DenseMatrix[Double]]): Double = {
      val thetaKernel = if (layer.bias) theta.head(::, 1 until theta.head.cols) else theta.head
      val kernelPenalty = sum(DiffFunction[KR].apply(layer.kernelReg, thetaKernel))
      val biasPenalty = if (layer.bias) {
        DiffFunction[KR].apply(layer.kernelReg, theta.head(::, 0))
      } else {
        0.0
      }
      kernelPenalty + biasPenalty
    }

    override def backward(layer: Dense[A, KR, BR], theta: List[DenseMatrix[Double]], input: DenseMatrix[Double], error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]]) = {
      val inputB = if (layer.bias) DenseMatrix.horzcat(DenseMatrix.ones[Double](input.rows, 1), input) else input
      val grad = layer.activation.gradient1(inputB * theta.head.t) // M x OUT
      val delta: DenseMatrix[Double] = grad *:* error // M x OUT
      (delta, List(delta.t * inputB))
    }

    override def backwardPenalty(layer: Dense[A, KR, BR], theta: List[DenseMatrix[Double]]): List[DenseMatrix[Double]] = {
      val thetaKernel = if (layer.bias) theta.head(::, 1 until theta.head.cols) else theta.head
      val kernelPenalty = DiffFunction[KR].gradient(layer.kernelReg, thetaKernel)
      val biasPenalty = if (layer.bias) {
        DiffFunction[KR].gradient(layer.kernelReg, theta.head(::, 0))
      } else {
        DenseVector.zeros[Double](theta.head.rows)
      }
      List(DenseMatrix.horzcat(biasPenalty.toDenseMatrix.t, kernelPenalty))
    }
  }
}
