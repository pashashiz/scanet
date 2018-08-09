package io.scanet.optimizers

trait OptimizerInst
  extends SGDInst
    with AdaGradInst
    with AdaDeltaInst
    with RMSPropInst
    with AdamInst
    with AdamaxInst
    with NadamInst {
}
