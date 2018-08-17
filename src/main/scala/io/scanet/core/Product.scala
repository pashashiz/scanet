package io.scanet.core

import simulacrum.{op, typeclass}

@typeclass trait Product[A] {

  @op("|+|", alias = true)
  def product[B](a: A, b: B): (A, B) = (a, b)
}

