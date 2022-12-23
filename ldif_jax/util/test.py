import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import sys
import os
from camera_util import *

ldif_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ldif_root)
print(ldif_root)
import ldif.ldif.util.camera_util as camera_util_tf
import unittest

class TestCameraUtil(unittest.TestCase):
    
    def test_roll_pitch_yaw_to_rotation_matrices(self):
        x = np.array([np.pi/3.0, np.pi/4.0, np.pi/6.0])
        result_jax = roll_pitch_yaw_to_rotation_matrices(x)
        result_tf = camera_util_tf.roll_pitch_yaw_to_rotation_matrices(x)
#        print(result_tf)
        self.assertEqual(tf.convert_to_tensor(result_jax, dtype='float64'), result_tf)

    def test_look_at_np(self):
        eye = np.array(np.random.rand(1,3))*100
        center = np.array(np.random.rand(1,3))*100
        world_up = np.array(np.random.rand(1,3))*100
        result_jax = look_at_np(eye, center, world_up)
        result_tf = camera_util_tf.look_at_np(eye, center, world_up)
        self.assertEqual(result_jax, result_tf)



"""
batch_size = 100
# coordinates between 0 and 100
eye = jnp.array(np.random.rand(batch_size, 3))*100
center = jnp.array(np.random.rand(batch_size, 3))*100
world_up = jnp.array(np.random.rand(batch_size, 3))*100
"""

eye = jnp.array(np.random.rand(1,3))*100
center = jnp.array(np.random.rand(1,3))*100
world_up = jnp.array(np.random.rand(1,3))*100
print("TEST LOOK_AT_NP")
print(look_at_np(eye, center, world_up))
print("TF: ")
print(camera_util_tf.look_at_np(eye, center, world_up))

print("TEST_ROLL_PITCH_YAW_TO_ROTATION_MATRICES")
x = np.array([np.pi/3.0, np.pi/4.0, np.pi/6.0])
print(roll_pitch_yaw_to_rotation_matrices(x))
print("TF: ")
tf.print(camera_util_tf.roll_pitch_yaw_to_rotation_matrices(x))


if __name__ == '__main__':
    unittest.main()
