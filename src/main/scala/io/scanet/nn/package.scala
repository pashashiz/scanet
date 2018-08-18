package io.scanet

import breeze.linalg._
import breeze.stats._
import io.scanet.func.DiffFunction.DFBuilder

package object nn {

  case class Scaled(shift: Double, scale: Double)

  def normalize(data: DenseMatrix[Double]): (Scaled, DenseMatrix[Double]) = {
    val shift = mean(data)
    val scale = stddev(data)
    (Scaled(shift, scale), (data - shift) / scale)
  }


  def nnError[A: Layer](layer: A, out: DenseMatrix[Double]): DFBuilder[NNError[A]] =
    coef => NNError(layer, coef, out)

  def nn[A: Layer](layer: A)(theta: DenseVector[Double]): NN[A] = {
    NN(layer, theta)
  }

}
