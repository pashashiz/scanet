package io.scanet.func

import breeze.linalg.{*, DenseMatrix, DenseVector}
import io.scanet.func.DiffFunction.DFBuilder
import simulacrum.{op, typeclass}

@typeclass trait Product[A] {

  @op("|+|", alias = true)
  def product[B](a: A, b: B): (A, B) = (a, b)
}

@typeclass trait ProductK[F[_]] {
  @op("<+>", alias = true)
  def productK[A, B](fa: F[A], fb: F[B]): F[(A, B)]
}


@typeclass trait Function[A] extends Product[A] {

  def apply(f: A , vars: DenseVector[Double]): Double

  def apply(f: A , vars: DenseMatrix[Double]): DenseVector[Double] =
    vars(*, ::).map(apply(f, _))
}

@typeclass trait DiffFunction[A] extends Function[A] {

  def gradient(f: A , vars: DenseVector[Double]): DenseVector[Double]

  def gradient(f: A , vars: DenseMatrix[Double]): DenseMatrix[Double] =
    vars(*, ::).map(gradient(f, _))
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

trait FunctionsSyntax
  extends Function.ToFunctionOps
    with DiffFunction.ToDiffFunctionOps
    with ProductK.ToProductKOps
    with Product.ToProductOps