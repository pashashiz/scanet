package io.scanet.nn

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.pow
import io.scanet.func.DiffFunction
import io.scanet.func.DiffFunction.DFBuilder

case class NNError[A: Layer](layer: A, in: DenseMatrix[Double], out: DenseMatrix[Double])

trait NNErrorFunctionInst extends Layer.ToLayerOps {

  implicit def NNErrorFunctionInst[A: Layer]: DiffFunction[NNError[A]] = new DiffFunction[NNError[A]] {

    override def gradient(f: NNError[A], vars: DenseVector[Double]): DenseVector[Double] = {
      var theta = f.layer.unpack(f.in.cols)(vars)
      var error =  f.layer.forward(theta, f.in) - f.out
      var (_, grad) = f.layer.backprop(theta, f.in, error)
      f.layer.pack(grad)
    }

    override def apply(f: NNError[A], vars: DenseVector[Double]): Double = {
      var theta = f.layer.unpack(f.in.cols)(vars)
      var realOut = f.layer.forward(theta, f.in)
      val error = realOut - f.out
      val errorPerLabel = error(::, *).map(column => {
        val squaredErrors: DenseVector[Double] = pow(column, 2)
        sum(squaredErrors) / (2 * column.length)
      })
      sum(errorPerLabel)
    }
  }
}
