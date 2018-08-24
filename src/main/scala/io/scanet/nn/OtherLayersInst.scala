package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.nn.Layer.ops._


trait OtherLayersInst {

  implicit def TupleLayer[A: Layer, B: Layer]: Layer[(A, B)] = new Layer[(A, B)] {

    override def forward(layer: (A, B), theta: List[DenseMatrix[Double]], input: DenseMatrix[Double]): DenseMatrix[Double] = {
      val (left, right) = layer
      val leftTheta = theta.take(left.power)
      val rightTheta = theta.slice(left.power, left.power + right.power)
      right.forward(rightTheta, left.forward(leftTheta, input))
    }

    override def backprop(layer: (A, B), theta: List[DenseMatrix[Double]], input: DenseMatrix[Double],
                          error: DenseMatrix[Double]): (DenseMatrix[Double], List[DenseMatrix[Double]]) = {
      val (left, right) = layer
      val leftTheta = theta.take(left.power) // OUT_l x IN
      val rightInput = left.forward(leftTheta, input) // M x OUT_r
      val rightTheta = theta.slice(left.power, left.power + right.power) // OUT_r x OUT_l
      val (rightDelta, rightGrad) = right.backprop(rightTheta, rightInput, error) // M x OUT_r, OUT_r x OUT_l
      val rightThetaNB = if (right.shape.head.bias) rightTheta.head(::, 1 until rightTheta.head.cols) else rightTheta.head
      val leftError = rightDelta * rightThetaNB // M x OUT_r * OUT_r x OUT_l = M x OUT_l
      val (leftDelta, leftGrad) = left.backprop(leftTheta, input, leftError) // OUT_l x IN
      (leftDelta, leftGrad ++ rightGrad)
    }

    override def power(layer: (A, B)): Int = layer match {
      case (left, right) => left.power + right.power
    }

    override def shape(layer: (A, B)): List[Shape] = layer match {
      case (left, right) => left.shape ++ right.shape
    }
  }
}


