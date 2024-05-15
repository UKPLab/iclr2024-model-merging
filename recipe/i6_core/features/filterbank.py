__all__ = ["FilterbankJob", "filterbank_flow", "filter_width_from_channels"]

import copy

import numpy as np

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def FilterbankJob(crp, filterbank_options=None, **kwargs):
    """
    :param rasr.crp.CommonRasrParameters crp:
    :param dict[str, Any]|None filterbank_options:
    :return: Feature extraction job with filterbank flow
    :rtype: FeatureExtractionJob
    """
    if filterbank_options is None:
        filterbank_options = {}
    else:
        filterbank_options = copy.deepcopy(filterbank_options)

    if "samples_options" not in filterbank_options:
        filterbank_options["samples_options"] = {}
    filterbank_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = filterbank_flow(**filterbank_options)

    port_name_mapping = {"features": "fb"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="filterbank",
        **kwargs,
    )


def filterbank_flow(
    warping_function="mel",
    filter_width=70,
    normalize=True,
    normalization_options=None,
    without_samples=False,
    samples_options=None,
    fft_options=None,
    apply_log=False,
    add_epsilon=False,
    add_features_output=False,
):
    """

    :param str warping_function: "mel" or "bark"
    :param int filter_width: filter width in Hz. Please use :func:`filter_width_from_channels` to get N filters.
    :param bool normalize: add a final signal-normalization node
    :param dict[str, Any]|None normalization_options: option dict for `signal-normalization` flow node
    :param bool without_samples: creates the flow network without a sample flow, but expects "samples" as input
    :param dict[str, Any]|None samples_options: parameter dict for :func:`samples_flow`
    :param dict[str, Any]|None fft_options: parameter dict for :func:`fft_flow`
    :param bool apply_log: adds a logarithm before normalization
    :param bool add_epsilon: if a logarithm should be applied, add a small epsilon to prohibit zeros
    :param bool add_features_output: Add the output port "features". This should be set to True,
        default is False to not break existing hash.
    :return: filterbank flow network
    :rtype: rasr.FlowNetwork
    """
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

    filterbank = net.add_node(
        "signal-filterbank",
        "filterbank",
        {"warping-function": warping_function, "filter-width": filter_width},
    )
    net.link(
        fft_mapping[fft_net.get_output_links("amplitude-spectrum").pop()], filterbank
    )

    if apply_log:
        if add_epsilon:
            nonlinear = net.add_node(
                "generic-vector-f32-log-plus", "nonlinear", {"value": "1.175494e-38"}
            )
        else:
            nonlinear = net.add_node("generic-vector-f32-log", "nonlinear")
        net.link(filterbank, nonlinear)
        filterbank_out = nonlinear
    else:
        filterbank_out = filterbank

    if normalize:
        attr = {
            "type": "mean-and-variance",
            "length": "infinity",
            "right": "infinity",
        }
        attr.update(normalization_options)
        normalization = net.add_node(
            "signal-normalization", "filterbank-normalization", attr
        )
        net.link(filterbank_out, normalization)
        net.link(normalization, "network:features")
    else:
        net.link(filterbank_out, "network:features")

    return net


def filter_width_from_channels(channels, warping_function="mel", f_max=8000, f_min=0):
    """
    Per default we use FilterBank::stretchToCover, it computes it number of filters:
      number_of_filters = (maximumFrequency_ - minimumFrequency_ - filterWidth_) / spacing_ + 1));
    :param int channels: Number of channels of the filterbank
    :param str warping_function: Warping function used by the filterbank. ['mel', 'bark']
    :param float f_max: Filters are placed only below this frequency in Hz.
        The physical maximum is half of the audio sample rate, but lower values make possibly more sense.
    :param float f_min: Filters are placed only over this frequency in Hz
    :return: filter-width
    :rtype float
    """

    if warping_function == "mel":
        warp = _mel_warping_function
    elif warping_function == "bark":
        warp = _bark_warping_function
    else:
        raise NotImplementedError

    maximumFrequency = warp(f_max)
    minimumFrequency = warp(f_min)
    targetfilterWidth = (maximumFrequency - minimumFrequency) / (
        ((channels - 1) / 2) + 1
    )
    return targetfilterWidth


def _mel_warping_function(f):
    return 2595 * np.log10(1 + f / 700)


def _bark_warping_function(f):
    return 6 * np.ln(f / 600 + np.sqrt((f / 600) ** 2 + 1))
