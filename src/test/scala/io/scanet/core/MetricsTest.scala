package io.scanet.core

import breeze.linalg.DenseMatrix
import io.scanet.test.CustomMatchers
import io.scanet.core.metrics._
import org.scalatest.FlatSpec

class MetricsTest extends FlatSpec with CustomMatchers {

  "binary accuracy" should "work" in {
    val expected = DenseMatrix(
      (1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
      (1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0))
    val actual = DenseMatrix(
      (0.0, 1.0, 0.0),
      (0.0, 1.0, 0.0),
      (1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0))
    binaryAccuracy(expected, actual) should be(0.75)
  }

}
