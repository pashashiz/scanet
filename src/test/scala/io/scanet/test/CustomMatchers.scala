package io.scanet.test

import java.io.File

import breeze.linalg.DenseVector
import org.scalatest.Matchers
import org.scalatest.matchers.{MatchResult, Matcher}
import breeze.linalg._
import breeze.numerics._

trait CustomMatchers extends Matchers {

  def beWithinTolerance(mean: Double, tolerance: Double) =
    be >= (mean - tolerance) and be <= (mean + tolerance)

  def beWithinTolerance(mean: DenseVector[Double], tolerance: Double): Matcher[DenseVector[Double]] = {
    left: DenseVector[Double] => {
      val diff = abs(left - mean)
      val results: BitVector = diff <:< DenseVector.fill(mean.length, tolerance)
      val matches = results.reduce(_ && _)
      MatchResult(matches, s"$left does not equal to $mean +/-$tolerance", "")
    }
  }

  def beWithinTolerance(mean: DenseMatrix[Double], tolerance: Double): Matcher[DenseMatrix[Double]] = {
    left: DenseMatrix[Double] => {
      val diff = abs(left - mean)
      val results = diff <:< DenseMatrix.fill(mean.rows, mean.cols)(tolerance)
      val matches = results(*, ::).map(_.reduce(_ && _)).reduce(_ && _)
      MatchResult(matches, s"$left does not equal to $mean +/-$tolerance", "")
    }
  }

  def resource(name: String): File = {
    new File(Thread.currentThread().getContextClassLoader.getResource(name).getFile)
  }
}
