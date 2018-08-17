package io.scanet.core

import simulacrum.{op, typeclass}

@typeclass trait ProductK[F[_]] {

  @op("<+>", alias = true)
  def productK[A, B](fa: F[A], fb: F[B]): F[(A, B)]
}


