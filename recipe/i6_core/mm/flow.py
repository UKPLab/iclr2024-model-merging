__all__ = [
    "linear_segmentation_flow",
    "alignment_flow",
    "cached_alignment_flow",
    "dump_alignment_flow",
    "confidence_based_alignment_flow",
]

from i6_core.rasr import FlowNetwork, RasrConfig


def linear_segmentation_flow(feature_energy_net, alignment_cache=None):
    assert all(
        out in feature_energy_net.get_output_ports() for out in ["features", "energy"]
    )

    net = FlowNetwork()
    net.add_output("alignments")
    net.add_param(["id", "orthography", "TASK"])

    mapping = net.add_net(feature_energy_net)
    net.interconnect_inputs(feature_energy_net, mapping)
    net.interconnect_outputs(feature_energy_net, mapping)

    alignment = net.add_node(
        "speech-linear-segmentation",
        "alignment",
        {"id": "$(id)", "orthography": "$(orthography)"},
    )
    net.link(mapping[feature_energy_net.get_output_links("energy").pop()], alignment)

    if alignment_cache is None:
        net.link(alignment, "network:alignments")
    else:
        cache = net.add_node(
            "generic-cache", "alignment-cache", {"id": "$(id)", "path": alignment_cache}
        )
        net.link(alignment, cache)
        net.link(cache, "network:alignments")

    return net


def alignment_flow(feature_net, alignment_cache_path=None):
    assert "features" in feature_net.get_output_ports()
    net = FlowNetwork()
    net.add_output("alignments")
    net.add_param(["id", "orthography", "TASK"])

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    aggregate = net.add_node("generic-aggregation-vector-f32", "aggregate")
    net.link(mapping[feature_net.get_output_links("features").pop()], aggregate)

    alignment = net.add_node(
        "speech-alignment",
        "alignment",
        {"id": "$(id)", "orthography": "$(orthography)"},
    )
    net.link(aggregate, alignment)

    if alignment_cache_path is not None:
        cache = net.add_node(
            "generic-cache",
            "alignment-cache",
            {"id": "$(id)", "path": alignment_cache_path},
        )
        net.link(alignment, cache)
        net.link(cache, "network:alignments")
    else:
        net.link(alignment, "network:alignments")

    return net


def cached_alignment_flow(feature_net, alignment_cache_path):
    assert "features" in feature_net.get_output_ports()
    net = FlowNetwork()
    net.add_output("alignments")
    net.add_param(["id", "TASK"])

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    cache = net.add_node(
        "generic-cache",
        "alignment-cache",
        {"id": "$(id)", "path": alignment_cache_path},
    )
    net.link(cache, "network:alignments")

    return net


def dump_alignment_flow(feature_net, original_alignment, new_alignment):
    assert "features" in feature_net.get_output_ports()
    net = FlowNetwork()
    net.add_output("alignments")
    net.add_param("orthography")

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    cache = net.add_node(
        "generic-cache", "alignment-cache", {"id": "$(id)", "path": original_alignment}
    )

    aggregate = net.add_node("generic-aggregation-vector-f32", "aggregate")
    net.link(mapping[feature_net.get_output_links("features").pop()], aggregate)

    dumper = net.add_node(
        "speech-alignment-dump", "dumper", {"id": "$(id)", "file": new_alignment}
    )
    net.link(cache, dumper)
    net.link(aggregate, "%s:features" % dumper)
    net.link(dumper, "network:alignments")

    return net


def confidence_based_alignment_flow(
    feature_net,
    lattice_cache_path,
    alignment_cache_path=None,
    global_scale=1.0,
    confidence_threshold=0.75,
    weight_scale=1.0,
    ref_alignment_path=None,
):
    assert "features" in feature_net.get_output_ports()
    net = FlowNetwork()
    net.add_output("alignments")
    net.add_param(["id", "orthography", "TASK"])

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)
    net.interconnect_outputs(feature_net, mapping)

    net.config = RasrConfig()

    aggregate = net.add_node("generic-aggregation-vector-f32", "aggregate")
    net.link(mapping[feature_net.get_output_links("features").pop()], aggregate)

    model_comb = net.add_node("model-combination", "model-combination")
    net.config[model_comb].mode = "complete"

    lattice_reader = net.add_node("lattice-read", "lattice-reader", {"id": "$(id)"})
    net.config[lattice_reader].readers = "acoustic, lm"
    net.config[lattice_reader].acoustic_scale = 1.0
    net.config[lattice_reader].lm_scale = 1.0
    net.config[lattice_reader].lattice_archive.path = lattice_cache_path
    net.link(model_comb, lattice_reader)

    lattice_wp_sr = net.add_node("lattice-semiring", "lattice-wp-semiring")
    net.config[lattice_wp_sr].type = "log"
    net.config[lattice_wp_sr].keys = "acoustic lm"
    net.config[lattice_wp_sr].acoustic.scale = global_scale
    net.config[lattice_wp_sr].lm.scale = global_scale
    net.link(lattice_reader, lattice_wp_sr)

    lattice_wp = net.add_node("lattice-word-posterior", "lattice-wp")
    net.link(lattice_wp_sr, lattice_wp)

    lattice_wp_p_sr = net.add_node("lattice-semiring", "lattice-wp-post-semiring")
    net.config[lattice_wp_p_sr].type = "log"
    net.config[lattice_wp_p_sr].keys = "confidence"
    net.config[lattice_wp_p_sr].confidence.scale = 1.0
    net.link(lattice_wp, lattice_wp_p_sr)

    lattice_expm = net.add_node("lattice-expm", "lattice-expm")
    net.link(lattice_wp_p_sr, lattice_expm)

    alignment = net.add_node(
        "speech-alignment-from-lattice", "alignment", {"id": "$(id)"}
    )
    net.link(lattice_expm, alignment)
    net.link(aggregate, alignment + ":features")
    net.link(model_comb, alignment + ":model-combination")

    if ref_alignment_path is None:
        lattice_best = net.add_node("lattice-nbest", "lattice-best")
        net.link(lattice_wp_sr, lattice_best)

        alignment_best = net.add_node(
            "speech-alignment-from-lattice", "alignment-best", {"id": "$(id)"}
        )
        net.link(lattice_best, alignment_best)
        net.link(aggregate, alignment_best + ":features")
        net.link(model_comb, alignment_best + ":model-combination")
    else:
        alignment_best = net.add_node(
            "generic-cache",
            "alignment-cache",
            {"id": "$(id)", "path": ref_alignment_path},
        )

    alignment_state_conf = net.add_node(
        "alignment-weights-by-tied-state-alignment-weights",
        "alignment-state-confidence",
    )
    net.link(alignment, alignment_state_conf)
    net.link(alignment_best, alignment_state_conf + ":target")

    reset_large_weights = net.add_node(
        "speech-alignment-reset-weights",
        "alignment-reset-large-weights",
        {
            "mode": "larger-than",
            "previous-weight": confidence_threshold,
            "new-weight": 1.0,
        },
    )
    net.link(alignment_state_conf, reset_large_weights)

    reset_small_weights = net.add_node(
        "speech-alignment-reset-weights",
        "alignment-reset-small-weights",
        {"mode": "smaller-than", "previous-weight": 0.99, "new-weight": 0.0},
    )
    net.link(reset_large_weights, reset_small_weights)

    multiply_weights = net.add_node(
        "speech-alignment-multiply-weights",
        "alignment-multiply-weights",
        {"factor": weight_scale},
    )
    net.link(reset_small_weights, multiply_weights)

    if alignment_cache_path is not None:
        cache = net.add_node(
            "generic-cache",
            "alignment-cache",
            {"id": "$(id)", "path": alignment_cache_path},
        )
        net.link(multiply_weights, cache)
        net.link(cache, "network:alignments")
    else:
        net.link(multiply_weights, "network:alignments")

    return net
