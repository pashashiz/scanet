package io.scanet.func

import breeze.linalg.{*, DenseMatrix, DenseVector}
import io.scanet.core._
import io.scanet.func.DiffFunction.DFBuilder
import simulacrum.typeclass


@typeclass trait Function[A] extends Product[A] {

  def apply(f: A , vars: DenseVector[Double]): Double

  def apply(f: A , vars: DenseMatrix[Double]): DenseVector[Double] =
    vars(*, ::).map(apply(f, _))

  def apply1(f: A, vars1: Double): Double =
    apply(f, DenseVector(vars1))

  def apply1(f: A, vars1: DenseVector[Double]): DenseVector[Double] =
    apply(f, DenseMatrix(vars1))

  def apply1(f: A, vars1: DenseMatrix[Double]): DenseMatrix[Double] =
    vars1(*, ::).map(apply1(f, _))

}

@typeclass trait DiffFunction[A] extends Function[A] {

  def gradient(f: A , vars: DenseVector[Double]): DenseVector[Double]

  def gradient(f: A , vars: DenseMatrix[Double]): DenseMatrix[Double] =
    vars(*, ::).map(gradient(f, _))

  def gradient1(f: A , vars1: Double): Double =
    gradient(f, DenseVector(vars1))(0)

  def gradient1(f: A , vars1: DenseVector[Double]): DenseVector[Double] =
    gradient(f, DenseMatrix(vars1))(::, 0)

  def gradient1(f: A , vars1: DenseMatrix[Double]): DenseMatrix[Double] =
    vars1(*, ::).map(gradient1(f, _))
}

object DiffFunction {

  type DFBuilder[B] = DenseMatrix[Double] => B

}

trait DefaultFunctionsInst {

  implicit def tupleDiffFunctionXInst[A: DiffFunction, B: DiffFunction]: DiffFunction[(A, B)] = new DiffFunction[(A, B)] {

    override def gradient(f: (A, B), vars: DenseVector[Double]): DenseVector[Double] =
      DiffFunction[A].gradient(f._1, vars) + DiffFunction[B].gradient(f._2, vars)

    override def apply(f: (A, B), vars: DenseVector[Double]): Double =
      DiffFunction[A].apply(f._1, vars) + DiffFunction[B].apply(f._2, vars)
  }

  implicit def DFBuilderProductK: ProductK[DFBuilder] = new ProductK[DFBuilder] {
    override def productK[A, B](fa: DFBuilder[A], fb: DFBuilder[B]): DFBuilder[(A, B)] =
      coef => (fa(coef), fb(coef))
  }
}
