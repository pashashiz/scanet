package io.scanet.nn

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.pow
import io.scanet.func.{DiffFunction, splitXY}
import simulacrum.typeclass

// Layer can be converted to diff function
// SGD.optimize((Dense(1, 2) :: Dense(1, 2)).toDFBuilder[A])
// or with ToDFBuilder[A]Opt
// SGD.optimize(Dense(1, 2) :: Dense(1, 2))

// we will have a method append "::" to compose 2 layers and as a result we will get Layer[Composed]

@typeclass
trait Layer[A] {

  def forward(layer: A, coef: DenseVector[Double], vars: DenseMatrix[Double]): DenseVector[Double]

//  def toDFBuilder[A](layer: A): DFBuilder[A] =
//    coef => new DiffFunction {
//
//      override def apply(vars: DenseMatrix[Double]): Double = {
//        val (xs, y) = splitXY(vars)
//        def result = forward(layer, coef, xs)
//        0.5/(vars.rows: Double) * sum(pow(result - y, 2))
//      }
//
//      override def gradient(vars: DenseMatrix[Double]): DenseVector[Double] = ???
//    }


}
