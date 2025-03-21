#!/bin/sh
python vanilla_gan.py --data_preprocess=basic
python vanilla_gan.py --data_preprocess=deluxe
python vanilla_gan.py --data_preprocess=basic --use_diffaug
python vanilla_gan.py --data_preprocess=deluxe --use_diffaug