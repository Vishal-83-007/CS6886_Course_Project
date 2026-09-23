"""
Input encoding -- faithful port of the main loop in
Diehl&Cook_spiking_MNIST.py, specifically:

    rates = image.reshape(n_input) / 8. * input_intensity
    input_groups['Xe'].rate = rates
    b.run(single_example_time)
    current_spike_count = spike_counters['Ae'].count - previous_spike_count
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        ... re-run with rate=0 for resting_time, then retry with the
            same image at the new (higher) input_intensity ...
    else:
        ... record result, proceed to next image ...
        input_intensity = start_input_intensity   # reset to 2. for next image

NOT the paper prose's "+32 Hz" boost -- that was this project's earlier
paper-text-based guess. The actual code scales the WHOLE rate vector by
a multiplicative `input_intensity` factor (starting at 2.0, +1 per
retry), applied on top of the fixed `pixel / 8` base rate.
"""

import numpy as np
from brian2 import Hz, ms, second

SINGLE_EXAMPLE_TIME = 0.35 * second   # source: single_example_time
RESTING_TIME = 0.15 * second           # source: resting_time
START_INPUT_INTENSITY = 2.0             # source: start_input_intensity
MIN_SPIKES = 5                           # source: hardcoded `< 5` check


def image_to_rates(image_flat_0_255, input_intensity):
    """source: rates = image.reshape(n_input) / 8. * input_intensity"""
    return (image_flat_0_255 / 8.0) * input_intensity * Hz


def present_example(components, image_flat_0_255, learning_on=True):
    """
    Present one image with the exact retry/intensity-boost logic from
    the original main loop. Returns the excitatory spike-count vector
    from the (eventually) successful presentation.
    """
    from brian2 import SpikeMonitor
    from .network import set_plasticity

    net = components['net']
    input_group = components['input']
    exc_group = components['exc']
    n_e = components['n_e']

    set_plasticity(components, learning_on)

    mon = SpikeMonitor(exc_group, name='tmp_exc_mon')
    net.add(mon)

    input_intensity = START_INPUT_INTENSITY
    counts = np.zeros(n_e)
    try:
        while True:
            input_group.rates = image_to_rates(image_flat_0_255, input_intensity)
            n_before = mon.num_spikes
            net.run(SINGLE_EXAMPLE_TIME)
            n_after = mon.num_spikes
            spikes_this_run = n_after - n_before

            if spikes_this_run >= MIN_SPIKES:
                if spikes_this_run:
                    idx = mon.i[-spikes_this_run:]
                    counts = np.bincount(idx, minlength=n_e).astype(float)
                # source: rate=0, run(resting_time), THEN reset intensity
                input_group.rates = 0 * Hz
                net.run(RESTING_TIME)
                break
            else:
                # source: input_intensity += 1; rate=0; run(resting_time); retry
                input_intensity += 1
                input_group.rates = 0 * Hz
                net.run(RESTING_TIME)
    finally:
        net.remove(mon)

    return counts
