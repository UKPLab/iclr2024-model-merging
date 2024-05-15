__all__ = ["VoicedJob", "voiced_flow"]

from .common import *
from .extraction import *
import i6_core.rasr as rasr


def VoicedJob(crp, voiced_options=None, **kwargs):
    if voiced_options is None:
        voiced_options = {}
    else:
        voiced_options = copy.deepcopy(voiced_options)
    if "samples_options" not in voiced_options:
        voiced_options["samples_options"] = {}
    voiced_options["samples_options"]["audio_format"] = crp.audio_format
    feature_flow = voiced_flow(**voiced_options)

    port_name_mapping = {"voiced": "voiced"}

    return FeatureExtractionJob(
        crp,
        feature_flow,
        port_name_mapping,
        job_name="Energy",
        **kwargs,
    )


def voiced_flow(
    window_shift=0.01,
    window_duration=0.04,
    min_pos=0.0025,
    max_pos=0.0167,
    without_samples=False,
    samples_options={},
    add_voiced_output=False,
):
    net = rasr.FlowNetwork()
    if add_voiced_output:
        net.add_output("voiced")

    if without_samples:
        net.add_input("samples")
        samples = "network:samples"
    else:
        samples_net = samples_flow(**samples_options)
        samples_mapping = net.add_net(samples_net)
        samples = samples_mapping[samples_net.get_output_links("samples").pop()]

    win = net.add_node(
        "signal-window",
        "voiced-window",
        {"type": "rectangular", "shift": window_shift, "length": window_duration},
    )
    net.link(samples, win)

    pad = net.add_node(
        "signal-vector-f32-resize", "padded-window", {"new-size": window_duration}
    )
    net.link(win, pad)

    norm = net.add_node(
        "signal-vector-f32-mean-energy-normalization", "voiced-normalization"
    )
    net.link(pad, norm)

    autocorrelation = net.add_node(
        "signal-cross-correlation",
        "voiced-autocorrelation",
        {
            "begin": 0.0,
            "end": window_duration,
            "normalization": "unbiased-estimate",
            "similarity-function": "multiplication",
            "use-fft": True,
        },
    )
    net.link(norm, autocorrelation + ":x")
    net.link(norm, autocorrelation + ":y")

    peak = net.add_node(
        "signal-peak-detection",
        "voiced-peak-detection",
        {"min-position": min_pos, "max-position": max_pos},
    )
    net.link(autocorrelation, peak)

    convert = net.add_node("generic-convert-f32-to-vector-f32", "convert")
    net.link(peak + ":maximal-peak-value", convert)
    net.link(convert, "network:voiced")

    return net
