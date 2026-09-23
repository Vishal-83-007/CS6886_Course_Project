"""
Diehl & Cook (2015) spiking neural network for MNIST digit recognition.

Architecture (Fig. 1/2 of Diehl & Cook, matches ASP paper's baseline topology):
    Input (784, Poisson)  --STDP, all-to-all-->  Excitatory (N)
    Excitatory (N)        --fixed, one-to-one-->  Inhibitory (N)
    Inhibitory (N)        --fixed, all-to-all-except-self-->  Excitatory (N)

Neuron model: conductance-based Leaky-Integrate-and-Fire (LIF).
Learning rule: power-law weight-dependent STDP (Diehl & Cook Eqn. 3),
triggered only on postsynaptic spikes, using a single presynaptic trace
with a target-trace offset for depression of uncorrelated synapses:

    delta_w = eta * (x_pre - x_tar) * (w_max - w)^mu

Homeostasis: adaptive membrane threshold v_thresh_e + theta, where theta
increases by theta_plus on every excitatory spike and decays exponentially
with a long time constant (Diehl & Cook Sec. 2.4).

This module intentionally does NOT include ASP's continuous weight-decay
mechanism -- see README.md for why the baseline is trained with plain
Diehl & Cook STDP.
"""

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, ms, mV, Hz, second,
    Network, defaultclock, seed as brian_seed
)

# ----------------------------------------------------------------------
# Neuron parameters (Diehl & Cook, Sec. 2.1 / 2.2)
# ----------------------------------------------------------------------
V_REST_E = -65 * mV
V_REST_I = -60 * mV
V_RESET_E = -65 * mV
V_RESET_I = -45 * mV
V_THRESH_E = -52 * mV
V_THRESH_I = -40 * mV
REFRAC_E = 5 * ms
REFRAC_I = 2 * ms

TAU_E = 100 * ms   # excitatory membrane tau (increased from bio ~10-20ms
                    # to 100ms, per D&C Sec 2.1, to better estimate rate code)
TAU_I = 10 * ms     # inhibitory membrane tau
TAU_GE = 1 * ms     # excitatory conductance decay
TAU_GI = 2 * ms     # inhibitory conductance decay

# Homeostasis (adaptive threshold), Sec. 2.4
THETA_PLUS_E = 0.05 * mV
TAU_THETA = 1e7 * ms     # very slow decay -> near-permanent within a run
THETA_OFFSET = 20 * mV
EXC_SPIKE_INTERVAL = 50 * ms  # 5 ms voltage clamp, 50 ms spike exclusion

# ----------------------------------------------------------------------
# STDP parameters (Diehl & Cook Eqn. 3)
# ----------------------------------------------------------------------
TC_PRE = 20 * ms      # presynaptic trace time constant
X_TAR = 0.4            # target presynaptic trace value (depression offset)
MU = 0.9                # weight-dependence exponent
ETA = 0.0001             # learning rate
W_MAX_EE = 1.0            # max input->excitatory weight (normalized)

# Fixed weights for the inhibitory circuit
W_EI = 10.4    # excitatory -> inhibitory (one-to-one), strong: 1 spike -> 1 spike
W_IE = 17.0    # inhibitory -> excitatory (all-to-all-except-self)

# ----------------------------------------------------------------------
# Equations
# ----------------------------------------------------------------------

EXC_EQS = '''
dv/dt = ((v_rest_e - v) + ge * (0*mV - v) + gi * (-100*mV - v)) / tau_e : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
dtheta/dt = -theta * plasticity_enabled / tau_theta : volt
dtimer/dt = 1 : second
plasticity_enabled : 1 (shared)
v_rest_e : volt
tau_e : second
tau_ge : second
tau_gi : second
tau_theta : second
'''

INH_EQS = '''
dv/dt = ((v_rest_i - v) + ge * (0*mV - v) + gi * (-85*mV - v)) / tau_i : volt (unless refractory)
dge/dt = -ge / tau_ge : 1
dgi/dt = -gi / tau_gi : 1
v_rest_i : volt
tau_i : second
tau_ge : second
tau_gi : second
'''

STDP_EQS = '''
w : 1
dpretrace/dt = -pretrace / tau_tr : 1 (event-driven)
tau_tr : second
x_tar : 1
mu : 1
eta : 1
w_max : 1
'''

STDP_ON_PRE = '''
ge_post += w
pretrace += 1
'''

STDP_ON_POST = '''
w = clip(w + eta * (pretrace - x_tar) * (w_max - w)**mu, 0, w_max)
'''


def build_network(n_excitatory=400, seed=0):
    """
    Build the Diehl & Cook three-layer SNN.

    Returns a dict with the Brian2 objects (net, groups, synapses) so the
    caller can control training/testing phases (toggle plasticity, read
    out weights, etc.).
    """
    rng = np.random.default_rng(seed)
    brian_seed(seed)  # Also seed Brian's Poisson spikes and synaptic randomness.
    n_input = 28 * 28
    n_e = n_excitatory
    n_i = n_excitatory

    defaultclock.dt = 0.5 * ms

    # ---- Input layer: Poisson spike generators (rates set per-example) ----
    from brian2 import PoissonGroup
    input_group = PoissonGroup(n_input, rates=0 * Hz)

    # ---- Excitatory layer ----
    exc_group = NeuronGroup(
        n_e, EXC_EQS,
        threshold='v > (theta - theta_offset + v_thresh_e) and timer > spike_interval',
        refractory='refrac_e',
        reset='v = v_reset_e; theta += theta_plus_e * plasticity_enabled; timer = 0*ms',
        method='euler', name='excitatory',
        namespace={'v_thresh_e': V_THRESH_E, 'v_reset_e': V_RESET_E,
                   'theta_offset': THETA_OFFSET, 'theta_plus_e': THETA_PLUS_E,
                   'refrac_e': REFRAC_E, 'spike_interval': EXC_SPIKE_INTERVAL}
    )
    exc_group.v = V_REST_E - 40 * mV
    exc_group.v_rest_e = V_REST_E
    exc_group.tau_e = TAU_E
    exc_group.tau_ge = TAU_GE
    exc_group.tau_gi = TAU_GI
    exc_group.tau_theta = TAU_THETA
    exc_group.theta = 20 * mV
    exc_group.timer = EXC_SPIKE_INTERVAL + defaultclock.dt
    exc_group.plasticity_enabled = 1

    # ---- Inhibitory layer ----
    inh_group = NeuronGroup(
        n_i, INH_EQS, threshold='v > v_thresh_i',
        refractory=REFRAC_I, reset='v = v_reset_i',
        method='euler', name='inhibitory',
        namespace={'v_thresh_i': V_THRESH_I, 'v_reset_i': V_RESET_I}
    )
    inh_group.v = V_REST_I - 40 * mV
    inh_group.v_rest_i = V_REST_I
    inh_group.tau_i = TAU_I
    inh_group.tau_ge = TAU_GE
    inh_group.tau_gi = TAU_GI

    # ---- Input -> Excitatory: all-to-all, plastic (STDP) ----
    syn_input_exc = Synapses(
        input_group, exc_group, model=STDP_EQS,
        on_pre=STDP_ON_PRE, on_post=STDP_ON_POST,
        method='euler', name='input_to_exc'
    )
    syn_input_exc.connect()  # all-to-all
    syn_input_exc.w = rng.uniform(0.0, 0.3, size=len(syn_input_exc.w))
    syn_input_exc.tau_tr = TC_PRE
    syn_input_exc.x_tar = X_TAR
    syn_input_exc.mu = MU
    syn_input_exc.eta = ETA
    syn_input_exc.w_max = W_MAX_EE

    # ---- Excitatory -> Inhibitory: one-to-one, fixed ----
    syn_exc_inh = Synapses(exc_group, inh_group, model='w : 1',
                            on_pre='ge_post += w', name='exc_to_inh')
    syn_exc_inh.connect(j='i')
    syn_exc_inh.w = W_EI

    # ---- Inhibitory -> Excitatory: all-to-all except self, fixed ----
    syn_inh_exc = Synapses(inh_group, exc_group, model='w : 1',
                            on_pre='gi_post += w', name='inh_to_exc')
    syn_inh_exc.connect(condition='i != j')
    syn_inh_exc.w = W_IE

    net = Network(input_group, exc_group, inh_group,
                   syn_input_exc, syn_exc_inh, syn_inh_exc)
    exc_monitor = SpikeMonitor(exc_group, record=False, name='exc_count_monitor')
    net.add(exc_monitor)

    return {
        'net': net,
        'input': input_group,
        'exc': exc_group,
        'inh': inh_group,
        'syn_input_exc': syn_input_exc,
        'syn_exc_inh': syn_exc_inh,
        'syn_inh_exc': syn_inh_exc,
        'exc_monitor': exc_monitor,
        '_learning_on': True,
        'n_e': n_e,
        'n_input': n_input,
    }


def set_plasticity(components, learning_on: bool):
    """Freeze both synaptic weights and adaptive thresholds for observation."""
    if components['_learning_on'] == learning_on:
        return
    components['syn_input_exc'].eta = ETA if learning_on else 0.0
    components['exc'].plasticity_enabled = int(learning_on)
    components['_learning_on'] = learning_on
