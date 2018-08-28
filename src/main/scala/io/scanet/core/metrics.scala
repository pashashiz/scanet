package io.scanet.core

import breeze.linalg._
import breeze.numerics._

object metrics {

  def binaryAccuracy(expected: DenseMatrix[Double], predicted: DenseMatrix[Double]): Double = {
    var diff: DenseMatrix[Long] = abs(round(expected) - round(predicted))
    var result = sum(diff(*, ::)).map(el => if (el == 0) 1 else 0)
    var all = result.length
    var positive = sum(result)
    positive.toDouble/all.toDouble
  }

}
