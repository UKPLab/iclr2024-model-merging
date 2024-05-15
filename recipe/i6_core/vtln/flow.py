__all__ = [
    "add_static_warping_to_filterbank_flow",
    "warp_filterbank_with_map_flow",
    "label_features_with_map_flow",
    "recognized_warping_factor_flow",
]

import i6_core.rasr as rasr


def add_static_warping_to_filterbank_flow(
    feature_net,
    alpha_name="warping-alpha",
    omega_name="warping-omega",
    node_name="filterbank",
):
    assert node_name in feature_net.nodes
    assert feature_net.nodes[node_name]["filter"] == "signal-filterbank"

    # copy original net
    net = rasr.FlowNetwork(name=feature_net.name)
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    net.add_param([alpha_name, omega_name])
    node = net.nodes[mapping[node_name]]
    node["warping-function"] = "nest(linear-2($(%s), $(%s)), %s)" % (
        alpha_name,
        omega_name,
        node["warping-function"],
    )

    return net


def warp_filterbank_with_map_flow(
    feature_net,
    map_file,
    map_key="$(id)",
    default_output=1.0,
    omega=0.875,
    node_name="filterbank",
):
    assert node_name in feature_net.nodes
    assert feature_net.nodes[node_name]["filter"] == "signal-filterbank"

    # copy original net
    net = rasr.FlowNetwork()
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    node = net.nodes[mapping[node_name]]
    node["warping-function"] = "nest(linear-2($input(alpha), %s), %s)" % (
        omega,
        node["warping-function"],
    )

    corpus_map = net.add_node(
        "generic-coprus-key-map",
        "warping-factor",
        {
            "key": map_key,
            "map-file": map_file,
            "default-output": "%s" % default_output,
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
        },
    )
    net.link(corpus_map, "%s:alpha" % mapping[node_name])

    return net


def label_features_with_map_flow(
    feature_net, map_file, map_key="$(id)", default_output=1.0
):
    # copy original net
    net = rasr.FlowNetwork(name=feature_net.name)
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    if map_key.startswith("$(") and map_key.endswith(")"):
        net.add_param(map_key[2:-1])

    net.add_output("labels")
    corpus_map = net.add_node(
        "generic-coprus-key-map",
        "warping-factor",
        {
            "key": map_key,
            "map-file": map_file,
            "default-output": "%s" % default_output,
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
        },
    )
    net.link(corpus_map, "network:labels")

    return net


def recognized_warping_factor_flow(
    feature_net,
    alphas_file,
    mixtures,
    filterbank_node="filterbank",
    amplitude_spectrum_node="amplitude-spectrum",
    omega=0.875,
):
    assert filterbank_node in feature_net.nodes
    assert feature_net.nodes[filterbank_node]["filter"] == "signal-filterbank"
    assert amplitude_spectrum_node in feature_net.nodes

    # copy original net
    net = rasr.FlowNetwork(name=feature_net.name)
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    # remove output for features
    original_feature_outputs = net.get_output_links("features")
    net.unlink(to_name="%s:%s" % (net.name, "features"))

    warped_net, broken_links = feature_net.subnet_from_node(filterbank_node)

    warped_mapping = net.add_net(warped_net)
    net.interconnect_outputs(warped_net, warped_mapping)

    for l in broken_links:
        net.link(mapping[l[0]], warped_mapping[l[1]])

    fbnode = net.nodes[warped_mapping[filterbank_node]]
    fbnode["warping-function"] = "nest(linear-2($input(alpha), %s), %s)" % (
        omega,
        fbnode["warping-function"],
    )

    # energy
    energy = net.add_node("generic-vector-f32-norm", "energy", {"value": 1})
    net.link(mapping[amplitude_spectrum_node], energy)

    convert_energy_to_vector = net.add_node(
        "generic-convert-f32-to-vector-f32", "convert-energy-to-vector"
    )
    net.link(energy, convert_energy_to_vector)

    energy_normalization = net.add_node(
        "signal-normalization",
        "energy-normalization",
        {"type": "divide-by-mean", "length": "infinite", "right": "infinite"},
    )
    net.link(convert_energy_to_vector, energy_normalization)

    convert_energy_to_scalar = net.add_node(
        "generic-convert-vector-f32-to-f32", "convert-energy-vector-to-scalar"
    )
    net.link(energy_normalization, convert_energy_to_scalar)

    energy_sync = net.add_node("generic-synchronization", "energy-sync")
    net.link(convert_energy_to_scalar, energy_sync)
    net.link(original_feature_outputs.pop(), "%s:target" % energy_sync)

    rec = net.add_node(
        "signal-bayes-classification",
        "warping-factor-recognizer",
        {"class-label-file": alphas_file},
    )
    net.link(rec, "%s:alpha" % warped_mapping[filterbank_node])
    net.link(energy_sync, "%s:feature-score-weight" % rec)
    net.link("%s:target" % energy_sync, rec)

    net.config = rasr.RasrConfig()
    net.config[rec].likelihood_function.file = mixtures
    net.config[rec].likelihood_function.feature_scorer_type = "SIMD-diagonal-maximum"

    return net
