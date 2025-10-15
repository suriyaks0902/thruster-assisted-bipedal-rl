import tensorflow as tf
import numpy as np
import gym
from typing import Tuple, List, Optional
import logging
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

# Enable TensorFlow 1.x compatibility mode for existing code
tf.compat.v1.disable_eager_execution()

# Configure GPU for TensorFlow 1.x compatibility mode
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Configure GPU memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Set GPU as default device for all operations
    tf.config.set_visible_devices(gpus[0], 'GPU')
    # Force all operations to GPU
    tf.config.set_soft_device_placement(False)
    print(f"✅ Policy GPU configured: {gpus[0]}")
    print("🚀 Forcing all operations to GPU (no CPU fallback)")
else:
    print("⚠️ No GPU available for policies")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPCNNPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.compat.v1.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.compat.v1.get_variable_scope().name

    def _init(
        self,
        ob_space_vf,
        ob_space_pol,
        ob_space_pol_cnn,
        ac_space,
        hid_size,
        num_hid_layers,
        gaussian_fixed_var=True,
    ):
        # assert isinstance(ob_space_vf, gym.spaces.Box)
        # assert isinstance(ob_space_pol, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None
        shape_cnn = [ob_space_pol_cnn[0], ob_space_pol_cnn[1], 1]

        ob_vf = U.get_placeholder(
            name="ob_vf",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space_vf.shape),
        )
        ob_pol = U.get_placeholder(
            name="ob_pol",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space_pol.shape),
        )
        ob_pol_cnn = U.get_placeholder(
            name="ob_pol_cnn",
            dtype=tf.float32,
            shape=[sequence_length] + list(shape_cnn),
        )

        with tf.compat.v1.variable_scope("obfilter_vf"):
            self.ob_vf_rms = RunningMeanStd(shape=ob_space_vf.shape)
        with tf.compat.v1.variable_scope("obfilter_pol"):
            self.ob_pol_rms = RunningMeanStd(shape=ob_space_pol.shape)
        with tf.compat.v1.variable_scope("obfilter_pol_cnn"):
            self.ob_pol_cnn_rms = RunningMeanStd(shape=shape_cnn)

        with tf.compat.v1.variable_scope("vf"):
            obz = tf.clip_by_value(
                (ob_vf - self.ob_vf_rms.mean) / self.ob_vf_rms.std, -5.0, 5.0
            )
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        hid_size,
                        name="fc%i" % (i + 1),
                        kernel_initializer=U.normc_initializer(1.0),
                    )
                )
            self.vpred = tf.compat.v1.layers.dense(
                last_out, 1, name="final", kernel_initializer=U.normc_initializer(1.0)
            )[:, 0]

        with tf.compat.v1.variable_scope("pol"):
            ob_cnn = tf.clip_by_value(
                (ob_pol_cnn - self.ob_pol_cnn_rms.mean) / self.ob_pol_cnn_rms.std + 1,
                -5.0,
                5.0,
            )
            x = tf.nn.relu(
                U.conv2d(
                    ob_cnn, 32, "l1", [ob_space_pol_cnn[0], 6], [1, 3], pad="VALID"
                )
            )
            x = tf.nn.relu(U.conv2d(x, 16, "l2", [1, 4], [1, 2], pad="VALID"))
            x = U.flattenallbut0(x)

            last_out = tf.clip_by_value(
                (ob_pol - self.ob_pol_rms.mean) / self.ob_pol_rms.std, -5.0, 5.0
            )
            last_out = tf.concat([last_out, x], axis=1)

            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        hid_size,
                        name="fc%i" % (i + 1),
                        kernel_initializer=U.normc_initializer(1.0),
                    )
                )
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        pdtype.param_shape()[0] // 2,
                        name="final_temp",
                        kernel_initializer=U.normc_initializer(0.01),
                    ),
                    name="final",
                )
                logstd = tf.compat.v1.get_variable(
                    name="logstd",
                    shape=[1, pdtype.param_shape()[0] // 2],
                    initializer=tf.constant_initializer(np.log(0.1)),
                    trainable=False,
                )
                pdparam = tf.concat(
                    [mean, mean * 0.0 + logstd], axis=1
                ) 
            else:
                pdparam = tf.compat.v1.layers.dense(
                    last_out,
                    pdtype.param_shape()[0],
                    name="final",
                    kernel_initializer=U.normc_initializer(0.01),
                )

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function(
            [stochastic, ob_vf, ob_pol, ob_pol_cnn], [ac, self.vpred]
        )
        self._act_pol = U.function([stochastic, ob_pol, ob_pol_cnn], [ac])
        self._act_lat = U.function([ob_pol, ob_pol_cnn], [x])

    def act(self, stochastic, ob_vf, ob_pol):
        ac1, vpred1 = self._act(
            stochastic, ob_vf[None], ob_pol[0][None], ob_pol[1][None]
        )
        return ac1[0], vpred1[0]

    def act_pol(self, stochastic, ob_pol):
        ac1 = self._act_pol(stochastic, ob_pol[0][None], ob_pol[1][None])
        return ac1[0]

    def get_latent(self, ob_pol):
        latent = self._act_lat(ob_pol[0][None], ob_pol[1][None])
        return latent[0]

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.compat.v1.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.compat.v1.get_variable_scope().name

    def _init(
        self,
        ob_space_vf,
        ob_space_pol,
        ac_space,
        hid_size,
        num_hid_layers,
        gaussian_fixed_var=True,
    ):
        assert isinstance(ob_space_vf, gym.spaces.Box)
        assert isinstance(ob_space_pol, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob_vf = U.get_placeholder(
            name="ob_vf",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space_vf.shape),
        )
        ob_pol = U.get_placeholder(
            name="ob_pol",
            dtype=tf.float32,
            shape=[sequence_length] + list(ob_space_pol.shape),
        )

        with tf.compat.v1.variable_scope("obfilter_vf"):
            self.ob_vf_rms = RunningMeanStd(shape=ob_space_vf.shape)
        with tf.compat.v1.variable_scope("obfilter_pol"):
            self.ob_pol_rms = RunningMeanStd(shape=ob_space_pol.shape)

        with tf.compat.v1.variable_scope("vf"):
            obz = tf.clip_by_value(
                (ob_vf - self.ob_vf_rms.mean) / self.ob_vf_rms.std, -5.0, 5.0
            )
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        hid_size,
                        name="fc%i" % (i + 1),
                        kernel_initializer=U.normc_initializer(1.0),
                    )
                )
            self.vpred = tf.compat.v1.layers.dense(
                last_out, 1, name="final", kernel_initializer=U.normc_initializer(1.0)
            )[:, 0]

        with tf.compat.v1.variable_scope("pol"):
            last_out = tf.clip_by_value(
                (ob_pol - self.ob_pol_rms.mean) / self.ob_pol_rms.std, -5.0, 5.0
            )
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        hid_size,
                        name="fc%i" % (i + 1),
                        kernel_initializer=U.normc_initializer(1.0),
                    )
                )
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                # mean = tf.compat.v1.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final_temp', kernel_initializer=U.normc_initializer(0.01))
                mean = tf.nn.tanh(
                    tf.compat.v1.layers.dense(
                        last_out,
                        pdtype.param_shape()[0] // 2,
                        name="final_temp",
                        kernel_initializer=U.normc_initializer(0.01),
                    ),
                    name="final",
                )
                logstd = tf.compat.v1.get_variable(
                    name="logstd",
                    shape=[1, pdtype.param_shape()[0] // 2],
                    initializer=tf.constant_initializer(np.log(0.1)),
                    trainable=False,
                )
                pdparam = tf.concat(
                    [mean, mean * 0.0 + logstd], axis=1
                )
            else:
                pdparam = tf.compat.v1.layers.dense(
                    last_out,
                    pdtype.param_shape()[0],
                    name="final",
                    kernel_initializer=U.normc_initializer(0.01),
                )

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob_vf, ob_pol], [ac, self.vpred])
        self._act_pol = U.function([stochastic, ob_pol], [ac])

    def act(self, stochastic, ob_vf, ob_pol):
        ac1, vpred1 = self._act(stochastic, ob_vf[None], ob_pol[None])
        return ac1[0], vpred1[0]

    def act_pol(self, stochastic, ob_pol):
        ac1 = self._act_pol(stochastic, ob_pol[None])
        return ac1[0]

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

# Probability distribution types (simplified versions)
class DiagGaussianPdType:
    def __init__(self, size: int):
        self.size = size
    
    def param_shape(self):
        return [self.size * 2]  # mean + logstd
    
    def pdfromflat(self, flat):
        return DiagGaussianPd(flat)


class CategoricalPdType:
    def __init__(self, ncat: int):
        self.ncat = ncat
    
    def param_shape(self):
        return [self.ncat]
    
    def pdfromflat(self, flat):
        return CategoricalPd(flat)


class DiagGaussianPd:
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(flat, 2, axis=-1)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    
    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    
    def mode(self):
        return self.mean


class CategoricalPd:
    def __init__(self, flat):
        self.flat = flat
        self.logits = flat
    
    def sample(self):
        return tf.random.categorical(self.logits, 1)
    
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

