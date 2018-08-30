package io.scanet

import breeze.linalg._
import breeze.stats._
import io.scanet.core.DFBuilder
import io.scanet.nn.Layer.ops._

package object nn {

  case class Scaled(shift: Double, scale: Double)

  def normalize(data: DenseMatrix[Double]): (Scaled, DenseMatrix[Double]) = {
    val shift = mean(data)
    val scale = stddev(data)
    (Scaled(shift, scale), (data - shift) / scale)
  }

  def normalize(data: DenseMatrix[Double], scaled: Scaled): DenseMatrix[Double] = {
    (data - scaled.shift) / scaled.scale
  }

  def nnError[A: Layer](layer: A): DFBuilder[NNError[A]] =
    coef => {
      val allUnits = coef.cols
      val outUnits = layer.shape.last.units
      val inUnits = allUnits - outUnits
      NNError(layer, coef(::, 0 until inUnits), coef(::, inUnits until allUnits))
    }

  def nn[A: Layer](layer: A)(theta: DenseVector[Double]): NN[A] = {
    NN(layer, theta)
  }

}
