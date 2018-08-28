package io.scanet.core

import breeze.linalg.{*, DenseMatrix, DenseVector}
import simulacrum.typeclass


@typeclass trait Function[A] extends Product[A] {

  def arity(f: A): Int = 1

  def apply(f: A , vars: DenseVector[Double]): Double

  def apply(f: A , vars: DenseMatrix[Double]): DenseVector[Double] =
    vars(*, ::).map(apply(f, _))

  def apply1(f: A, vars1: Double): Double =
    apply(f, DenseVector(vars1))

  def apply1(f: A, vars1: DenseVector[Double]): DenseVector[Double] =
    vars1.map(apply1(f, _))

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
    vars1.map(gradient1(f, _))

  def gradient1(f: A , vars1: DenseMatrix[Double]): DenseMatrix[Double] =
    vars1(*, ::).map(gradient1(f, _))
}

@typeclass trait FunctionM[A] extends Product[A] {

  def apply(f: A , vars: DenseVector[Double]): DenseVector[Double]

  def apply(f: A , vars: DenseMatrix[Double]): DenseMatrix[Double] =
    vars(*, ::).map(apply(f, _))

}



