"""
Faithful Brian2 port of Peter Diehl's original implementation:
    https://github.com/peter-u-diehl/stdp-mnist
    (Diehl&Cook_spiking_MNIST.py + Diehl&Cook_MNIST_random_conn_generator.py)

This is NOT a from-scratch reconstruction from the paper's prose. Every
equation, parameter, and mechanism below is taken directly from Diehl's
released code (verified by fetching both files), with exactly one
deliberate substitution, documented at the point it occurs: Brian 1's
`dtimer/dt = 100.0 : ms` idiom for a refractory countdown does not have a
literal Brian2 equivalent (it relies on Brian 1's internal unit-scaling
behavior), so it is re-expressed as `dtimer/dt = 1` with timer in seconds
-- functionally identical (elapsed real time since last reset), same
mechanism (threshold gated on a timer, NOT Brian2's native `refractory`
clause, so membrane potential keeps integrating -- and can keep being
pulled down by inhibition -- throughout what Diehl's code calls the
"refractory" window). This was the key structural difference identified
as the likely cause of the E/I oscillation instability found when the
project's first (paper-text-based) implementation used Brian2's native
`(unless refractory)` clause with these same conductance values.

Everything else -- neuron equations, STDP rule (the two-trace pre/post
rule, NOT the power-law rule from the paper's Eqn. 3), conductance
values, weight normalization target, initial weight distribution,
synaptic delays -- is copied directly from the source, not re-derived.
"""

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, ms, mV, Hz, second,
    Network, defaultclock
)

# ----------------------------------------------------------------------
# Parameters -- copied verbatim from Diehl&Cook_spiking_MNIST.py
# ----------------------------------------------------------------------
V_REST_E = -65 * mV
V_REST_I = -60 * mV
V_RESET_E = -65 * mV
V_RESET_I = -45 * mV
V_THRESH_E = -52 * mV
V_THRESH_I = -40 * mV
REFRAC_E = 5 * ms
REFRAC_I = 2 * ms

THETA_PLUS_E = 0.05 * mV
TAU_THETA = 1e7 * ms
THETA_OFFSET = 20.0 * mV   # `offset` in the original

# STDP (source: eqs_stdp_ee / eqs_stdp_pre_ee / eqs_stdp_post_ee)
TC_PRE_EE = 20 * ms
TC_POST_1_EE = 20 * ms
TC_POST_2_EE = 40 * ms
NU_EE_PRE = 0.0001    # learning rate, presynaptic-triggered depression
NU_EE_POST = 0.01       # learning rate, postsynaptic-triggered potentiation
WMAX_EE = 1.0

# Weight generation -- source: Diehl&Cook_MNIST_random_conn_generator.py
WEIGHT_EE_INPUT_INIT = 0.3   # initial input->excitatory weight scale
WEIGHT_EI = 10.4              # excitatory -> inhibitory (one-to-one)
WEIGHT_IE = 17.0               # inhibitory -> excitatory (all-to-all except self)

# Weight normalization target -- source: main script's `weight['ee_input'] = 78.`
WEIGHT_NORM_TARGET = 78.0

# Input->excitatory synaptic delay -- source: delay['ee_input'] = (0ms, 10ms)
DELAY_EE_INPUT_MIN = 0 * ms
DELAY_EE_INPUT_MAX = 10 * ms

# ----------------------------------------------------------------------
# Neuron equations -- copied verbatim (structurally) from neuron_eqs_e/i.
# The original scales through nS/amp intermediates
# (I_synE = ge*nS*(-v); dv/dt = (...+(I_synE+I_synI)/nS)/(100*ms)), which
# algebraically cancels to exactly ge*(-v) once nS is divided back out --
# so writing it directly as ge*(0*mV - v) below is the same computation,
# not a simplification that changes behavior (verified by hand:
# ge*nS*(-v) / nS = ge*(-v) identically).
# ----------------------------------------------------------------------

EXC_EQS = '''
dv/dt = ((v_rest_e - v) + ge*(0*mV - v) + gi*(-100*mV - v)) / (100*ms) : volt
dge/dt = -ge / (1*ms) : 1
dgi/dt = -gi / (2*ms) : 1
dtheta/dt = -theta / tau_theta : volt
dtimer/dt = 1 : second
v_rest_e : volt
tau_theta : second
'''

INH_EQS = '''
dv/dt = ((v_rest_i - v) + ge*(0*mV - v) + gi*(-85*mV - v)) / (10*ms) : volt
dge/dt = -ge / (1*ms) : 1
dgi/dt = -gi / (2*ms) : 1
v_rest_i : volt
'''

# Threshold: v > (theta - offset + v_thresh_e), gated by the manual timer
# (>= refrac_e) rather than Brian2's native `refractory` keyword -- this
# is the deliberate substitution described in the module docstring.
EXC_THRESHOLD = 'v > (theta - theta_offset + v_thresh_e) and (timer > refrac_e)'
EXC_RESET = 'v = v_reset_e; theta += theta_plus_e; timer = 0*second'

INH_THRESHOLD = 'v > v_thresh_i'
INH_RESET = 'v = v_reset_i'

# ----------------------------------------------------------------------
# STDP synapse model -- copied verbatim from eqs_stdp_ee / eqs_stdp_pre_ee
# / eqs_stdp_post_ee. This is the paper's "pre-and-post trace" rule
# (Eqn. 5-6 region of the paper text), NOT the power-law rule (Eqn. 3)
# used in this project's first implementation.
# ----------------------------------------------------------------------

STDP_EQS = '''
w : 1
post2before : 1
dpre/dt = -pre / tc_pre_ee : 1 (event-driven)
dpost1/dt = -post1 / tc_post_1_ee : 1 (event-driven)
dpost2/dt = -post2 / tc_post_2_ee : 1 (event-driven)
tc_pre_ee : second
tc_post_1_ee : second
tc_post_2_ee : second
nu_pre_ee : 1
nu_post_ee : 1
wmax_ee : 1
'''

# on presynaptic spike: deliver the synaptic conductance (this part isn't
# in eqs_stdp_pre_ee itself in the original -- it's handled by the
# separate `Connection` object with state='ge' that the STDP object
# wraps; folded in here since Brian2 Synapses combines both roles), then
# apply the original's exact STDP depression update.
STDP_ON_PRE = '''
ge_post += w
pre = 1.
w = clip(w - nu_pre_ee * post1, 0, wmax_ee)
'''

# on postsynaptic spike: copied verbatim from eqs_stdp_post_ee.
STDP_ON_POST = '''
post2before = post2
w = clip(w + nu_post_ee * pre * post2before, 0, wmax_ee)
post1 = 1.
post2 = 1.
'''


def build_network(n_excitatory=400, seed=0):
    """
    Build the network exactly as Diehl&Cook_spiking_MNIST.py constructs
    it (single population, n_i = n_e, dense connectivity).
    """
    rng = np.random.default_rng(seed)
    n_input = 28 * 28
    n_e = n_excitatory
    n_i = n_excitatory

    defaultclock.dt = 0.5 * ms   # source: Clock(dt=0.5*ms)

    # ---- Input layer ----
    input_group = PoissonGroup(n_input, rates=0 * Hz)

    # ---- Excitatory layer ----
    exc_group = NeuronGroup(
        n_e, EXC_EQS, threshold=EXC_THRESHOLD, reset=EXC_RESET,
        method='euler', name='excitatory',
        namespace={'v_thresh_e': V_THRESH_E, 'theta_offset': THETA_OFFSET,
                   'refrac_e': REFRAC_E, 'v_reset_e': V_RESET_E,
                   'theta_plus_e': THETA_PLUS_E}
    )
    exc_group.v = V_REST_E - 40 * mV     # source: v_rest_e - 40*mV
    exc_group.v_rest_e = V_REST_E
    exc_group.tau_theta = TAU_THETA
    exc_group.theta = THETA_OFFSET       # source: np.ones(n_e) * 20.0*mV
    exc_group.timer = 0 * second

    # ---- Inhibitory layer ----
    inh_group = NeuronGroup(
        n_i, INH_EQS, threshold=INH_THRESHOLD, refractory=REFRAC_I,
        reset=INH_RESET, method='euler', name='inhibitory',
        namespace={'v_thresh_i': V_THRESH_I, 'v_reset_i': V_RESET_I}
    )
    inh_group.v = V_REST_I - 40 * mV
    inh_group.v_rest_i = V_REST_I

    # ---- Input -> Excitatory: all-to-all, plastic, delayed ----
    syn_input_exc = Synapses(
        input_group, exc_group, model=STDP_EQS,
        on_pre=STDP_ON_PRE, on_post=STDP_ON_POST,
        method='euler', name='input_to_exc'
    )
    syn_input_exc.connect()
    # source: weightMatrix = (random(0,1) + 0.01) * weight['ee_input']
    syn_input_exc.w = (rng.uniform(0.0, 1.0, size=len(syn_input_exc.w)) + 0.01) * WEIGHT_EE_INPUT_INIT
    syn_input_exc.tc_pre_ee = TC_PRE_EE
    syn_input_exc.tc_post_1_ee = TC_POST_1_EE
    syn_input_exc.tc_post_2_ee = TC_POST_2_EE
    syn_input_exc.nu_pre_ee = NU_EE_PRE
    syn_input_exc.nu_post_ee = NU_EE_POST
    syn_input_exc.wmax_ee = WMAX_EE
    # source: delay['ee_input'] = (0ms, 10ms), uniform random per synapse
    syn_input_exc.delay = rng.uniform(float(DELAY_EE_INPUT_MIN), float(DELAY_EE_INPUT_MAX),
                                       size=len(syn_input_exc.w)) * second

    # ---- Excitatory -> Inhibitory: one-to-one, fixed, weight['ei']=10.4 ----
    syn_exc_inh = Synapses(exc_group, inh_group, model='w : 1',
                            on_pre='ge_post += w', name='exc_to_inh')
    syn_exc_inh.connect(j='i')
    syn_exc_inh.w = WEIGHT_EI

    # ---- Inhibitory -> Excitatory: all-to-all except self, weight['ie']=17.0 ----
    syn_inh_exc = Synapses(inh_group, exc_group, model='w : 1',
                            on_pre='gi_post += w', name='inh_to_exc')
    syn_inh_exc.connect(condition='i != j')
    syn_inh_exc.w = WEIGHT_IE

    net = Network(input_group, exc_group, inh_group,
                   syn_input_exc, syn_exc_inh, syn_inh_exc)

    return {
        'net': net,
        'input': input_group,
        'exc': exc_group,
        'inh': inh_group,
        'syn_input_exc': syn_input_exc,
        'syn_exc_inh': syn_exc_inh,
        'syn_inh_exc': syn_inh_exc,
        'n_e': n_e,
        'n_input': n_input,
    }


def normalize_weights(components, target_sum=WEIGHT_NORM_TARGET):
    """
    Verbatim port of the original's normalize_weights(): rescale each
    excitatory neuron's total incoming input weight to target_sum.
    """
    syn = components['syn_input_exc']
    n_e = components['n_e']
    w = np.array(syn.w[:])
    j_idx = np.array(syn.j[:])
    col_sums = np.bincount(j_idx, weights=w, minlength=n_e)
    col_sums[col_sums == 0] = 1.0
    col_factors = target_sum / col_sums
    syn.w = w * col_factors[j_idx]


def set_plasticity(components, learning_on: bool):
    """Turn STDP on/off (source: ee_STDP_on flag controls whether the
    STDP object is attached at all; here we just zero both learning
    rates, which is behaviorally equivalent)."""
    syn = components['syn_input_exc']
    if learning_on:
        syn.nu_pre_ee = NU_EE_PRE
        syn.nu_post_ee = NU_EE_POST
    else:
        syn.nu_pre_ee = 0.0
        syn.nu_post_ee = 0.0
