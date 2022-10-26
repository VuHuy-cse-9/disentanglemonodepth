# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from trainer_da import Trainer_da
from trainer_da_no_rgl import Trainer_da_no_grl
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.discrimination and opts.use_grl:
        trainer = Trainer_da(opts)
    elif opts.discrimination:
        trainer = Trainer_da_no_grl(opts)
    else:
        trainer = Trainer(opts)
    trainer.train()
