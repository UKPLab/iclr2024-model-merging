__all__ = ["MrastaJob", "mrasta_flow"]

import copy
import shutil

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def MrastaJob(crp, mrasta_options=None, **kwargs):
    if mrasta_options is None:
        mrasta_options = {}
    else:
        mrasta_options = copy.deepcopy(mrasta_options)
    if "samples_options" not in mrasta_options:
        mrasta_options["samples_options"] = {}
    mrasta_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = mrasta_flow(**mrasta_options)

    port_name_mapping = {"features-0": "high", "features-1": "low"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="Mrasta",
        **kwargs,
    )


def mrasta_flow(
    temporal_size=101,
    temporal_right=50,
    derivatives=1,
    gauss_filters=6,
    warping_function="mel",
    filter_width=268.258,
    filterbank_outputs=20,
    samples_options={},
    fft_options={},
):
    net = rasr.FlowNetwork()
    net.add_output(["features-0", "features-1"])
    samples_net = samples_flow(**samples_options)
    fft_net = fft_flow(**fft_options)

    samples_mapping = net.add_net(samples_net)
    net.interconnect_inputs(samples_net, samples_mapping)
    fft_mapping = net.add_net(fft_net)
    net.interconnect(samples_net, samples_mapping, fft_net, fft_mapping)

    filterbank = net.add_node(
        "signal-filterbank",
        "filterbank",
        {"warping-function": warping_function, "filter-width": filter_width},
    )
    net.link(
        fft_mapping[fft_net.get_output_links("amplitude-spectrum").pop()], filterbank
    )

    nonlinear = net.add_node("generic-vector-f32-log", "nonlinear")
    net.link(filterbank, nonlinear)

    # temporal context
    padding = net.add_node(
        "signal-vector-f32-sequence-concatenation",
        "window-padding",
        {
            "max-size": temporal_size,
            "right": temporal_right,
            "margin-condition": "present-not-empty",
            "margin-policy": "copy",
            "expand-timestamp": False,
        },
    )
    net.link(nonlinear, padding)

    # gaussian filter
    mrasta = net.add_node(
        "mrasta-filtering",
        "mrasta",
        {
            "context-length": temporal_size,
            "derivative": derivatives,
            "gauss-filter": gauss_filters,
        },
    )
    net.link(padding, mrasta)

    # normalization
    normalization = net.add_node(
        "signal-normalization",
        "normalization",
        {"length": "infinite", "right": "infinite", "type": "mean"},
    )
    net.link(mrasta, normalization)

    # The MRasta filter outputs 2 * #filters * ( #filterbank_outputs + (#filterbank_outputs - 2) * #derivatives )
    features_base = gauss_filters * filterbank_outputs
    features_derivative = gauss_filters * (filterbank_outputs - 2)
    ranges_high = [(0, features_base - 1)]
    ranges_low = [(features_base, 2 * features_base - 1)]
    for i in range(derivatives):
        ranges_high.append(
            (ranges_low[-1][1] + 1, ranges_low[-1][1] + features_derivative)
        )
        ranges_low.append(
            (ranges_high[-1][1] + 1, ranges_high[-1][1] + features_derivative)
        )

    select0 = net.add_node(
        "generic-vector-f32-select",
        "feature-selection-0",
        {"select": ",".join("-".join(map(str, r)) for r in ranges_high)},
    )
    net.link(normalization, select0)
    net.link(select0, "network:features-0")

    select1 = net.add_node(
        "generic-vector-f32-select",
        "feature-selection-1",
        {"select": ",".join("-".join(map(str, r)) for r in ranges_low)},
    )
    net.link(normalization, select1)
    net.link(select1, "network:features-1")

    return net
