import jax
import jax.numpy as jnp
import os
import sys
from sdf_util import *

import ndlib.models.ModelConfig as mc

ldif_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ldif_root)
print(ldif_root)
import ldif.ldif.util.sdf_util as sdf_util_tf
import unittest

#grads_and_vars = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
#print(clip_by_global_norm(grads_and_vars))

# Model Configuration
lhdn1 = 't'
hdn1 = 'test'
soft_transfer = True
offset = 2
hparams = {'lhdn': lhdn1, 'hdn': hdn1}
config = mc.Configuration()
#config.add_model_parameter('hparams', hparams)
config.hparams.update(hparams)

sdf = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1]])
print(apply_class_transfer(sdf, config, soft_transfer, offset))