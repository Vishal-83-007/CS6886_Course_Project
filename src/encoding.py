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
    from brian2 import SpikeMonitor

    net = components['net']
    input_group = components['input']
    exc_group = components['exc']

    mon = SpikeMonitor(exc_group, name='tmp_exc_mon')
    net.add(mon)

    rate_scale = 1.0
    counts = np.zeros(components['n_e'])
    try:
        for attempt in range(max_rate_scale_tries):
            input_group.rates = image_to_rates(image_flat_0_255, rate_scale)
            n_before = mon.num_spikes
            net.run(PRESENT_TIME)
            n_after = mon.num_spikes
            spikes_this_run = n_after - n_before

            if spikes_this_run >= MIN_SPIKES:
                counts = np.bincount(mon.i[-spikes_this_run:] if spikes_this_run else [],
                                      minlength=components['n_e']).astype(float)
                break
            else:
                rate_scale = rate_scale + RATE_BOOST_HZ / MAX_RATE_HZ
        # rest period
        input_group.rates = 0 * Hz
        net.run(REST_TIME)
    finally:
        net.remove(mon)

    return counts
