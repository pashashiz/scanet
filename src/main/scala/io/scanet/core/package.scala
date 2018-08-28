package io.scanet

import breeze.linalg.DenseMatrix

package object core {

  type DFBuilder[B] = DenseMatrix[Double] => B

}
