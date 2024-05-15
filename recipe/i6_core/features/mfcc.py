__all__ = ["MfccJob", "mfcc_flow"]

import copy
from typing import Any, Dict, Optional

from i6_core.features.common import cepstrum_flow, fft_flow, samples_flow
from i6_core.features.extraction import FeatureExtractionJob
from i6_core.rasr import FlowNetwork
from i6_core.rasr.crp import CommonRasrParameters


def MfccJob(
    crp: CommonRasrParameters, mfcc_options: Optional[Dict[str, Any]] = None, **kwargs
) -> FeatureExtractionJob:
    """
    :param crp:
    :param mfcc_options: Nested parameters for :func:`mfcc_flow`
    """
    if mfcc_options is None:
        mfcc_options = {}
    else:
        mfcc_options = copy.deepcopy(mfcc_options)

    if "samples_options" not in mfcc_options:
        mfcc_options["samples_options"] = {}
    mfcc_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = mfcc_flow(**mfcc_options)

    port_name_mapping = {"features": "mfcc"}

    return FeatureExtractionJob(
        crp, feature_flow, port_name_mapping, job_name="MFCC", **kwargs
    )


def mfcc_flow(
    warping_function: str = "mel",
    filter_width: float = 268.258,
    normalize: bool = True,
    normalization_options: Optional[Dict[str, Any]] = None,
    without_samples: bool = False,
    samples_options: Optional[Dict[str, Any]] = None,
    fft_options: Optional[Dict[str, Any]] = None,
    cepstrum_options: Optional[Dict[str, Any]] = None,
    add_features_output: bool = False,
) -> FlowNetwork:
    """
    :param warping_function:
    :param filter_width:
    :param normalize: whether to add or not a normalization layer
    :param normalization_options:
    :param without_samples:
    :param samples_options: arguments to :func:`~features.common.sample_flow`
    :param fft_options: arguments to :func:`~features.common.fft_flow`
    :param cepstrum_options: arguments to :func:`~features.common.cepstrum_flow`
    :param add_features_output: Add the output port "features" when normalize is True. This should be set to True,
        default is False to not break existing hash.
    """
    if normalization_options is None:
        normalization_options = {}
    if samples_options is None:
        samples_options = {}
    if fft_options is None:
        fft_options = {}
    if cepstrum_options is None:
        cepstrum_options = {}

    if normalize and "normalize" not in cepstrum_options:
        cepstrum_options["normalize"] = False

    net = FlowNetwork()

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

    cepstrum_net = cepstrum_flow(**cepstrum_options)
    cepstrum_mapping = net.add_net(cepstrum_net)
    for dst in cepstrum_net.get_input_links("in"):
        net.link(filterbank, cepstrum_mapping[dst])

    if normalize:
        attr = {
            "type": "mean-and-variance",
            "length": "infinity",
            "right": "infinity",
        }
        attr.update(normalization_options)
        normalization = net.add_node("signal-normalization", "mfcc-normalization", attr)
        for src in cepstrum_net.get_output_links("out"):
            net.link(cepstrum_mapping[src], normalization)
        if add_features_output:
            net.add_output("features")
        net.link(normalization, "network:features")
    else:
        net.interconnect_outputs(cepstrum_net, cepstrum_mapping, {"out": "features"})

    return net
