package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import io.scanet.func.FunctionM

case class NN[A: Layer](layer: A, theta: DenseVector[Double])

trait NNFunctionInst extends Layer.ToLayerOps {

  implicit def NNFunctionInst[A: Layer]: FunctionM[NN[A]] = new FunctionM[NN[A]] {
    override def apply(f: NN[A], vars: DenseVector[Double]): DenseVector[Double] =
      apply(f, vars.toDenseMatrix).toDenseVector

    override def apply(f: NN[A], vars: DenseMatrix[Double]): DenseMatrix[Double] =
      f.layer.forward(f.layer.unpack(vars.cols)(f.theta), vars)
  }
}
