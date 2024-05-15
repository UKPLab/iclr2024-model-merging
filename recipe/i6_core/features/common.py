__all__ = [
    "raw_audio_flow",
    "samples_flow",
    "feature_extraction_cache_flow",
    "basic_cache_flow",
    "external_file_feature_flow",
    "fft_flow",
    "cepstrum_flow",
    "add_derivatives",
    "add_linear_transform",
    "normalize_features",
    "make_first_feature_energy",
    "sync_energy_features",
    "sync_features",
    "select_features",
]

import i6_core.rasr as rasr

# -------------------- Helper -------------------


def get_input_node_type(audio_format="wav"):
    """
    Gets the RASR flow input node type based on the audio_format

    :param str audio_format:
    :return: node type
    :rtype: str
    """
    native_audio_formats = [
        "wav",
        "nist",
        "flac",
        "mpeg",
        "gsm",
        "htk",
        "phondat",
        "oss",
    ]
    return audio_format if audio_format in native_audio_formats else "ffmpeg"


# -------------------- Flows --------------------


def raw_audio_flow(audio_format="wav"):
    net = rasr.FlowNetwork()

    net.add_output("out")
    net.add_param(["input-file", "start-time", "end-time"])

    input_node_type = get_input_node_type(audio_format)

    samples = net.add_node(
        "audio-input-file-" + input_node_type,
        "samples",
        {
            "file": "$(input-file)",
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
        },
    )

    net.link(samples, "network:out")

    return net


def samples_flow(
    audio_format="wav",
    dc_detection=True,
    dc_params={
        "min-dc-length": 0.01,
        "max-dc-increment": 0.9,
        "min-non-dc-segment-length": 0.021,
    },
    input_options=None,
    scale_input=None,
):
    """
    Create a flow to read samples from audio files, convert it to f32 and apply optional dc-detection.

    Files that do not have a native input node will be opened with the ffmpeg flow node.
    Please check if scaling is needed.

    Native input formats are:
        - wav
        - nist
        - flac
        - mpeg (mp3)
        - gsm
        - htk
        - phondat
        - oss

    For more information see: https://www-i6.informatik.rwth-aachen.de/rwth-asr/manual/index.php/Audio_Nodes

    :param str audio_format: the input audio format
    :param bool dc_detection: enable dc-detection node
    :param dict dc_params: optional dc-detection node parameters
    :param dict input_options: additional options for the input node
    :param int|float|None scale_input: scale the waveform samples,
        this might be needed to scale ogg inputs by 2**15 to support feature flows
        designed for 16-bit wav inputs
    :return:
    """
    net = rasr.FlowNetwork()

    net.add_output("samples")
    net.add_param(["input-file", "start-time", "end-time", "track"])

    input_opts = {
        "file": "$(input-file)",
        "start-time": "$(start-time)",
        "end-time": "$(end-time)",
    }

    if input_options is not None:
        input_opts.update(**input_options)

    input_node_type = get_input_node_type(audio_format)

    samples = net.add_node("audio-input-file-" + input_node_type, "samples", input_opts)
    if input_node_type == "ffmpeg":
        samples_out = samples
    else:
        demultiplex = net.add_node(
            "generic-vector-s16-demultiplex", "demultiplex", track="$(track)"
        )
        net.link(samples, demultiplex)

        convert = net.add_node("generic-convert-vector-s16-to-vector-f32", "convert")
        net.link(demultiplex, convert)
        samples_out = convert

    if scale_input:
        scale = net.add_node(
            "generic-vector-f32-multiplication", "scale", value=str(scale_input)
        )
        net.link(samples_out, scale)
        pre_dc_out = scale
    else:
        pre_dc_out = samples_out

    if dc_detection:
        dc_detection = net.add_node("signal-dc-detection", "dc-detection", dc_params)
        net.link(pre_dc_out, dc_detection)
        net.link(dc_detection, "network:samples")
    else:
        net.link(pre_dc_out, "network:samples")

    return net


def feature_extraction_cache_flow(
    feature_net, port_name_mapping, one_dimensional_outputs=None
):
    """
    :param rasr.FlowNetwork feature_net: feature flow to extract features from
    :param dict[str,str] port_name_mapping: maps output ports to names of the cache files
    :param set[str]|None one_dimensional_outputs: output ports that return one-dimensional features (e.g. energy)
    :rtype: rasr.FlowNetwork
    """
    if one_dimensional_outputs is None:
        one_dimensional_outputs = set()

    net = rasr.FlowNetwork()

    net.add_output("features")
    net.add_param("id")
    net.add_param("TASK")
    node_mapping = net.add_net(feature_net)

    caches = []
    for port, name in port_name_mapping.items():
        node_name = "feature-cache-" + name
        fc = net.add_node(
            "generic-cache", node_name, {"id": "$(id)", "path": name + ".cache.$(TASK)"}
        )
        for src in feature_net.get_output_links(port):
            net.link(node_mapping[src], fc)

        if port in one_dimensional_outputs:
            convert = net.add_node(
                "generic-convert-f32-to-vector-f32", "convert-" + name
            )
            net.link(fc, convert)
            caches.append(convert)
        else:
            caches.append(fc)

    if len(caches) > 1:
        concat = net.add_node("generic-vector-f32-concat", "concat")
        for num, fc in enumerate(caches):
            net.link(fc, "%s:in%d" % (concat, num))
        net.link(concat, "network:features")
    else:
        net.link(caches[0], "network:features")

    return net


def basic_cache_flow(cache_files):
    if not type(cache_files) == list:
        cache_files = [cache_files]

    net = rasr.FlowNetwork()

    net.add_param("id")
    net.add_output("features")

    num_caches = len(cache_files)
    caches = []
    for num, cf in zip(_numerate(num_caches), cache_files):
        node_name = "cache" + num
        caches.append(
            net.add_node(
                "generic-cache",
                node_name,
                {"id": "$(id)", "path": rasr.NamedFlowAttribute(node_name, cf)},
            )
        )

    if len(caches) > 1:
        concat = net.add_node("generic-vector-f32-concat", "concat")
        for num, cache in enumerate(caches):
            net.link(cache, "concat:in%d" % num)
        net.link(concat, "network:features")
    else:
        net.link(caches[0], "network:features")

    return net


# FlowNetwork wrapper for an existing feature flow file
# Note: hard-coded output 'out'
def external_file_feature_flow(flow_file):
    net = rasr.FlowNetwork()

    net.add_param("input-file")
    net.add_param("start-time")
    net.add_param("end-time")
    net.add_param("track")
    net.add_param("id")
    net.add_output("features")

    bfe = net.add_node(
        flow_file,
        "base-feature-extraction",
        {
            "input-file": "$(input-file)",
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
            "track": "$(track)",
            "id": "$(id)",
            "ignore-unknown-parameters": "true",
        },
    )
    net.link(bfe + ":out", "network:features")
    return net


def fft_flow(
    preemphasis=1.0, window_type="hamming", window_shift=0.01, window_length=0.025
):
    net = rasr.FlowNetwork()

    net.add_input("samples")
    net.add_output("amplitude-spectrum")

    preemphasis = net.add_node("signal-preemphasis", "preemphasis", alpha=preemphasis)
    window = net.add_node(
        "signal-window",
        "window",
        {"type": window_type, "shift": window_shift, "length": window_length},
    )
    fft = net.add_node(
        "signal-real-fast-fourier-transform",
        "fft",
        {"maximum-input-size": window_length},
    )
    spectrum = net.add_node(
        "signal-vector-alternating-complex-f32-amplitude", "amplitude-spectrum"
    )

    net.link("network:samples", preemphasis)
    net.link(preemphasis, window)
    net.link(window, fft)
    net.link(fft, spectrum)
    net.link(spectrum, "network:amplitude-spectrum")

    return net


def cepstrum_flow(normalize=True, outputs=16, add_epsilon=False, epsilon=1.175494e-38):
    net = rasr.FlowNetwork()

    net.add_input("in")
    net.add_output("out")

    if add_epsilon:
        nonlinear = net.add_node(
            "generic-vector-f32-log-plus", "nonlinear", {"value": str(epsilon)}
        )
    else:
        nonlinear = net.add_node("generic-vector-f32-log", "nonlinear")
    cepstrum = net.add_node(
        "signal-cosine-transform", "cepstrum", {"nr-outputs": outputs}
    )

    net.link("network:in", nonlinear)
    net.link(nonlinear, cepstrum)

    if normalize:
        normalization = net.add_node(
            "signal-normalization",
            "normalization",
            {"length": "infinite", "right": "infinite", "type": "mean"},
        )
        net.link(cepstrum, normalization)
        net.link(normalization, "network:out")
    else:
        net.link(cepstrum, "network:out")

    return net


def add_derivatives(feature_net, derivatives=1):
    assert derivatives in [0, 1, 2]
    if derivatives == 0:
        return feature_net

    net = rasr.FlowNetwork()
    net.add_output("features")
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    delay = net.add_node(
        "signal-delay",
        "delay",
        {"max-size": 5, "right": 2, "margin-condition": "present-not-empty"},
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], delay)

    delta = net.add_node(
        "signal-regression", "delta", {"order": 1, "timestamp-port": 0}
    )
    for i in range(-2, 3):
        net.link("%s:%d" % (delay, i), "%s:%d" % (delta, i))

    if derivatives == 2:
        deltadelta = net.add_node(
            "signal-regression", "deltadelta", {"order": 2, "timestamp-port": 0}
        )
        for i in range(-2, 3):
            net.link("%s:%d" % (delay, i), "%s:%d" % (deltadelta, i))

    concat = net.add_node("generic-vector-f32-concat", "concat")
    net.link(
        mapping[feature_net.get_output_links("features").pop()], "%s:in-1" % concat
    )
    net.link(delta, "%s:in-2" % concat)
    if derivatives == 2:
        net.link(deltadelta, "%s:in-3" % concat)

    net.link(concat, "network:features")

    return net


def add_linear_transform(feature_net, matrix_path):
    net = rasr.FlowNetwork()
    net.add_output("features")

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    transform = net.add_node(
        "signal-matrix-multiplication-f32", "linear-transform", {"file": matrix_path}
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], transform)
    net.link(transform, "network:features")

    return net


def normalize_features(
    feature_net, length="infinite", right="infinite", norm_type="mean-and-variance"
):
    """
    Add normalization of the specfified type to the feature flow
    :param feature_net rasr.FlowNetwork: the unnormalized flow network, must have an output named 'features'
    :param length int|str: length of the normalization window in frames (or 'infinite')
    :param right int|str: number of frames right of the current position in the normalization window (can also be 'infinite')
    :param norm_type str: type of normalization, possible values are 'level', 'mean', 'mean-and-variance', 'mean-and-variance-1D', 'divide-by-mean', 'mean-norm'
    :returns rasr.FlowNetwork: input FlowNetwork with a signal-normalization node before the output
    """
    net = rasr.FlowNetwork()
    net.add_output("features")

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    normalization = net.add_node(
        "signal-normalization",
        "normalization",
        {"length": str(length), "right": str(right), "type": norm_type},
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], normalization)
    net.link(normalization, "network:features")

    return net


def make_first_feature_energy(feature_net):
    net = rasr.FlowNetwork()
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    net.add_output("energy")
    split = net.add_node("generic-vector-f32-split", "split")
    net.link(mapping[feature_net.get_output_links("features").pop()], split)
    convert = net.add_node("generic-convert-vector-f32-to-f32", "convert")
    net.link(split + ":0", convert)
    net.link(convert, "network:energy")

    return net


def sync_energy_features(feature_net, energy_net):
    assert "features" in feature_net.outputs
    assert "energy" in energy_net.outputs or "features" in energy_net.outputs
    energy_out = "energy" if "energy" in energy_net.outputs else "features"

    net = rasr.FlowNetwork()

    feature_mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, feature_mapping)
    net.interconnect_outputs(feature_net, feature_mapping)

    energy_mapping = net.add_net(energy_net)
    net.interconnect_inputs(energy_net, energy_mapping)

    sync = net.add_node("generic-synchronization", "energy-synchronization")
    net.link(
        feature_mapping[feature_net.get_output_links("features").pop()],
        sync + ":target",
    )
    net.link(energy_mapping[energy_net.get_output_links(energy_out).pop()], sync)

    net.add_output("energy")
    net.link(sync, "network:energy")

    return net


def sync_features(
    feature_net, target_net, feature_output="features", target_output="features"
):
    net = rasr.FlowNetwork()

    feature_mapping = net.add_net(feature_net)
    target_mapping = net.add_net(target_net)

    net.interconnect_inputs(feature_net, feature_mapping)
    net.interconnect_inputs(target_net, target_mapping)

    sync = net.add_node("signal-repeating-frame-prediction", "sync")
    net.link(feature_mapping[feature_net.get_output_links(feature_output).pop()], sync)
    net.link(
        target_mapping[target_net.get_output_links(target_output).pop()],
        sync + ":target",
    )

    net.add_output("features")
    net.link(sync, "network:features")

    return net


def select_features(feature_net, select_range):
    net = rasr.FlowNetwork()
    net.add_output("features")
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    select = net.add_node(
        "generic-vector-f32-select", "select", {"select": select_range}
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], select)
    net.link(select, "network:features")

    return net


# -------------------- Util --------------------


def _add_extension(file, ext):
    if not file.endswith("." + ext):
        return file + "." + ext
    return file


def _numerate(num, separator="-"):
    if num <= 1:
        return [""]
    else:
        return [separator + str(i) for i in range(num)]
