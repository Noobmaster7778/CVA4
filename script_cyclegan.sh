#!/bin/sh
 python cycle_gan.py --disc patch --train_iters 1000
 python cycle_gan.py --disc patch --use_cycle_consistency_loss  --train_iters 1000
 python cycle_gan.py --disc dc --use_cycle_consistency_loss

## More iterations
#python cycle_gan.py --disc patch --use_cycle_consistency_loss  --train_iters 10000