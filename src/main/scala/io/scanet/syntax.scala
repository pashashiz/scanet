package io.scanet

import io.scanet.func.{DefaultFunctionsInst, Function, DiffFunction, FunctionsInst, FunctionsSyntax}
import io.scanet.optimizers.{OptimizerInst, OptimizerSyntax}

object syntax
  extends OptimizerSyntax
     with OptimizerInst
     with DefaultFunctionsInst
     with FunctionsInst
     with FunctionsSyntax
     with cats.syntax.AllSyntax
     with cats.instances.AllInstances
//     with DiffFunction.ToDiffFunctionOps
//     with Function.ToFunctionOps
{}
