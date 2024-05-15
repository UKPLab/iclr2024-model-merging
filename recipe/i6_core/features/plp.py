__all__ = ["PlpJob", "plp_flow"]

import copy

from .common import *
from .extraction import *
from math import log, sqrt, floor
import i6_core.rasr as rasr


def PlpJob(crp, sampling_rate, plp_options=None, **kwargs):
    if plp_options is None:
        plp_options = {}
    else:
        plp_options = copy.deepcopy(plp_options)

    if "samples_options" not in plp_options:
        plp_options["samples_options"] = {}
    plp_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = plp_flow(sampling_rate=sampling_rate, **plp_options)

    port_name_mapping = {"features": "plp"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="PLP",
        **kwargs,
    )


def plp_flow(
    warping_function="bark",
    num_features=20,
    sampling_rate=8000,
    filter_width=3.8,
    normalize=True,
    normalization_options=None,
    without_samples=False,
    samples_options=None,
    fft_options=None,
    add_features_output=False,
):
    if normalization_options is None:
        normalization_options = {}
    if samples_options is None:
        samples_options = {}
    if fft_options is None:
        fft_options = {}

    net = rasr.FlowNetwork()
    if add_features_output:
        net.add_output("features")

    if without_samples:
        net.add_input("samples")
    else:
        samples_net = samples_flow(**samples_options)
        samples_mapping = net.add_net(samples_net)

    fft_net = fft_flow(**fft_options)
    fft_mapping = net.add_net(fft_net)

    if without_samples:
        net.interconnect_inputs(fft_net, fft_mapping)
    else:
        net.interconnect(samples_net, samples_mapping, fft_net, fft_mapping)

    power_spectrum = net.add_node(
        "generic-vector-f32-power", "power-spectrum", {"value": 2}
    )
    net.link(
        fft_mapping[fft_net.get_output_links("amplitude-spectrum").pop()],
        power_spectrum,
    )

    f = sampling_rate
    bark = 6 * log((f / 600) + sqrt((f / 600) ** 2 + 1))
    # For IncludeBoundary
    # Number of filters = floor((maximal-frequency - filter-width) / spacing + 1)
    # => spacing = (max-width) / num-1
    spacing = (bark - filter_width) / (num_features - 1)
    filterbank = net.add_node(
        "signal-filterbank",
        "filterbank",
        {
            "warping-function": warping_function,
            "filter-width": filter_width,
            "spacing": spacing,
            "type": "trapeze",
            "boundary": "include-boundary",
        },
    )
    net.link(power_spectrum, filterbank)

    split_filterbank = net.add_node("generic-vector-f32-split", "split-filterbank")
    net.link(filterbank, split_filterbank)

    reverse_split_filterbank = net.add_node(
        "generic-vector-f32-split", "reverse-split-filterbank", {"reverse": "true"}
    )
    net.link(filterbank, reverse_split_filterbank)

    copy_fl_filterbank = net.add_node(
        "generic-vector-f32-concat", "copy-first-last-filterbank"
    )
    net.link(split_filterbank + ":0", copy_fl_filterbank + ":first")
    net.link(filterbank, copy_fl_filterbank + ":middle")
    net.link(reverse_split_filterbank + ":0", copy_fl_filterbank + ":last")

    equal_loudness_preemphasis = net.add_node(
        "signal-vector-f32-continuous-transform",
        "equal-loudness-preemphasis",
        {
            "f": "nest(nest(disc-to-cont, invert(bark)), equal-loudness-preemphasis)",
            "operation": "multiplies",
        },
    )
    net.link(copy_fl_filterbank, equal_loudness_preemphasis)

    intensity_loudness_law = net.add_node(
        "generic-vector-f32-power", "intensity-loudness-law", {"value": "0.33"}
    )
    net.link(equal_loudness_preemphasis, intensity_loudness_law)

    autocorrelation = net.add_node(
        "signal-cosine-transform",
        "autocorrelation",
        {"nr-outputs": num_features, "input-type": "N-plus-one", "normalize": "true"},
    )
    net.link(intensity_loudness_law, autocorrelation)

    autoregression = net.add_node(
        "signal-autocorrelation-to-autoregression", "autoregression"
    )
    net.link(autocorrelation, autoregression)

    linear_cepstrum = net.add_node(
        "signal-autoregression-to-cepstrum",
        "linear-prediction-cepstrum",
        {"nr-outputs": num_features},
    )
    net.link(autoregression, linear_cepstrum)

    if normalize:
        attr = {
            "type": "mean-and-variance",
            "length": "infinity",
            "right": "infinity",
        }
        attr.update(normalization_options)
        normalization = net.add_node(
            "signal-normalization", "feature-normalization", attr
        )
        net.link(linear_cepstrum, normalization)
        net.link(normalization, "network:features")
    else:
        net.link(linear_cepstrum, "network:features")

    return net
