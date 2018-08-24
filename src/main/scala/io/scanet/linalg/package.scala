package io.scanet

import java.io.File

import breeze.linalg.DenseMatrix

package object linalg {

  /**
    * Split a matrix at `n` position
    *
    * @param n the position at which to split.
    * @param matrix matrix
    * @return a pair of matrix consisting of the first `n`
    *          elements of this matrix, and the other elements.
    */
  def splitColsAt[A](matrix: DenseMatrix[A], n: Int): (DenseMatrix[A], DenseMatrix[A]) = {
    require({n > 0}, "cannot split at 0 position ot less")
    require({n < matrix.cols}, s"a position should be  less than ${matrix.cols}")
    (matrix(::, 0 until n), matrix(::, n until n + 1))
  }

}
