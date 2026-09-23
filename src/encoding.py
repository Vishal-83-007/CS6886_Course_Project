"""
Poisson rate encoding of MNIST images (Diehl & Cook Sec. 2.5).

Pixel intensity (0-255) -> firing rate (0-63.75 Hz), i.e. intensity / 4.
Each example is presented for PRESENT_TIME, followed by REST_TIME of
zero input to let membrane/conductance variables relax.

If the excitatory layer fires fewer than MIN_SPIKES total during the
presentation, the input rates are boosted by RATE_BOOST and the SAME
example is re-presented (same procedure as the paper).
"""

import numpy as np
from brian2 import Hz, ms

from .network import set_plasticity

MAX_RATE_HZ = 63.75
PRESENT_TIME = 350 * ms
REST_TIME = 150 * ms
MIN_SPIKES = 5
RATE_BOOST_HZ = 32.0


def image_to_rates(image_flat_0_255, rate_scale=1.0):
    """image_flat_0_255: length-784 array, values in [0, 255]."""
    rates = (image_flat_0_255 / 4.0) * rate_scale
    return np.clip(rates, 0, None) * Hz


def present_example(components, image_flat_0_255, learning_on=True,
                     max_rate_scale_tries=6):
    """
    Present one MNIST image to the network, with automatic rate-boost
    re-presentation if too few excitatory spikes are produced.

    Returns: exc_spike_counts (n_e,) array of spikes fired during the
    (possibly boosted) successful presentation.
    """
    net = components['net']
    input_group = components['input']
    mon = components['exc_monitor']
    set_plasticity(components, learning_on)

    rate_scale = 1.0
    try:
        for attempt in range(max_rate_scale_tries):
            input_group.rates = image_to_rates(image_flat_0_255, rate_scale)
            before = mon.count[:]
            net.run(PRESENT_TIME)
            counts = mon.count[:] - before
            spikes_this_run = int(counts.sum())

            input_group.rates = 0 * Hz
            net.run(REST_TIME)

            if spikes_this_run >= MIN_SPIKES:
                break
            rate_scale += RATE_BOOST_HZ / MAX_RATE_HZ
    finally:
        input_group.rates = 0 * Hz

    return counts
