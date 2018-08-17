package io.scanet.func

import breeze.linalg._
import org.scalatest.FlatSpec
import io.scanet.syntax._

class NNIntro extends FlatSpec {

  "NN intro" should "show how simple NN works" in {
    /*

     a0     coef1      a1     coef2    a3

              ------- [x] -------\
    [x] -----/  /  /              \
               /  /   [x] -----\     \
    [x] ------/  /              \-----[x]
                /     [x] --------/ /
    [x] -------/                   /
                      [x] --------/

    [3x4]    [4x3]   [4x4]    [1x4]   [4x1]
     */

    // We have 3 features and 4 elements in our training set)
    val X = DenseMatrix(
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0),
      (1.0, 1.0, 1.0))
    // the output is just a one neuron
    val Y = DenseMatrix(
      0.0,
      1.0,
      1.0,
      0.0)

    // Layer 1 (4 neurons)
    val coef1 = DenseMatrix(
      (1.0, 0.1, 1.0),
      (0.5, 1.0, 0.0),
      (1.0, 1.0, 0.2),
      (0.1, 1.0, 0.3))
    // Layer 1 (1 neuron)
    val coef2 = DenseMatrix(
      (0.1, 0.5, 1.0, 0.0))

    // Forward pass
    val a0 = X
    val a1 = Sigmoid() apply1 a0 * coef1.t
    println(a1)
    val a2 = Sigmoid() apply1 a1 * coef2.t

    // Back-prop
    // a3
    val a2Error = Y - a2
    val a2Grad = Sigmoid() gradient1 a1 * coef2.t
    val a2Delta = a2Grad *:* a2Error
    val a2Diff = a2Delta.t * a1
    println(s"a2 diff:\n $a2Diff\n")
    // a2
    val a1Error = a2Delta * coef2
    val a1Grad = Sigmoid() gradient1 a0 * coef1.t
    val a1Delta = a1Grad *:* a1Error
    val a1Diff = a1Delta.t * a0
    println(s"a1 diff:\n $a1Diff\n")

  }

}
