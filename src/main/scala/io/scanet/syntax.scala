package io.scanet

import io.scanet.core.{CoreSyntax, FunctionsInst, FunctionsSyntax}
import io.scanet.nn.{NNInst, NNSyntax}
import io.scanet.optimizers.{OptimizerInst, OptimizerSyntax}

object syntax
  extends CoreSyntax
    with FunctionsInst
    with FunctionsSyntax
    with OptimizerSyntax
    with OptimizerInst
    with NNInst
    with NNSyntax
    with cats.syntax.AllSyntax
    with cats.instances.AllInstances {}
