package io.scanet.func

import breeze.linalg.DenseVector
import io.scanet.core.ProductK
import io.scanet.func.DiffFunction.DFBuilder

trait OtherFunctionsInst {

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
