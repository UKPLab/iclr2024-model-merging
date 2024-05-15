__all__ = ["add_cmllr_transform", "segment_clustering_flow"]

from sisyphus import tk

from i6_core.rasr import FlowNetwork


def add_cmllr_transform(
    feature_net: FlowNetwork,
    map_file: tk.Path,
    transform_dir: tk.Path,
    matrix_name: str = "$input(corpus-key).matrix",
) -> FlowNetwork:
    """

    :param feature_net: flow network for feature extraction, e.g. one from i6_core.features
    :param map_file: RASR corpus-key-map file, e.g. out_cluster_map_file from SegmentCorpusBySpeakerJob
    :param transform_dir: Directory containing the transformation matrix files, e.g. EstimateCMLLRJob.out_transforms
    :param matrix_name: Name pattern for the matrix files in the transform_dir
    :return: A new flow network with the CMLLR transformation added
    """
    net = FlowNetwork()
    net.add_param(["id", "start-time", "end-time"])
    net.add_output("features")

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    seg_clustering = net.add_node(
        "generic-coprus-key-map",
        "segment-clustering",
        {
            "key": "$(id)",
            "start-time": "$(start-time)",
            "end-time": "$(end-time)",
            "map-file": map_file,
        },
    )

    extend = net.add_node(
        "signal-vector-f32-resize",
        "extend",
        {
            "new-discrete-size": 1,
            "initial-value": 1.0,
            "relative-change": True,
            "change-front": True,
        },
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], extend)

    multiply = net.add_node(
        "signal-matrix-multiplication-f32",
        "apply-cmllr-transform",
        {
            "file": transform_dir.join_right(matrix_name),
        },
    )
    net.add_hidden_input(transform_dir)
    net.link(extend, multiply)
    net.link(seg_clustering, multiply + ":corpus-key")
    net.link(multiply, "network:features")

    return net


def segment_clustering_flow(
    feature_flow=None,
    file="cluster.map.$(TASK)",
    minframes=0,
    mincluster=2,
    maxcluster=100000,
    threshold=0,
    _lambda=1,
    minalpha=0.40,
    maxalpha=0.60,
    alpha=-1,
    amalgamation=0,
    infile=None,
    **kwargs,
):
    """
    :param feature_flow: Flownetwork of features used for clustering
    :param file: Name of the cluster outputfile
    :param minframes: minimum number of frames in a segment to consider the segment for clustering
    :param mincluster: minimum number of clusters
    :param maxcluster: maximum number of clusters
    :param threshold: Threshold for BIC which is added to the model-complexity based penalty
    :param _lambda: Weight for the model-complexity-based penalty (only lambda=1 corresponds to the definition of BIC; decreasing lambda increases the number of segment clusters.
    :param minalpha: Minimum Alpha scaling value used within distance scaling optimization
    :param maxalpha: Maximum Alpha scaling value used within distance scaling optimization
    :param alpha: Weighting Factor for correlation-based distance (default is automatic alpha estimation using minalpha and maxalpha values)
    :param amalgamation: Amalgamation Rule 1=Max Linkage, 0=Concatenation
    :param infile: Name of inputfile of clusters
    :return: (FlowNetwork)
    """
    net = FlowNetwork()
    net.add_output("features")
    net.add_param(["id", "TASK"])

    mapping = net.add_net(feature_flow)
    net.interconnect_inputs(feature_flow, mapping)

    cluster = net.add_node(
        "signal-segment-clustering",
        "segment-clustering-node",
        {
            "id": "$(id)",
            "file": file,
            "minframes": minframes,
            "mincluster": mincluster,
            "maxcluster": maxcluster,
            "threshold": threshold,
            "lambda": _lambda,
            "minalpha": minalpha,
            "maxalpha": maxalpha,
            "alpha": alpha,
            "amalgamation": amalgamation,
        },
    )

    net.link(mapping[feature_flow.get_output_links("features").pop()], cluster)
    net.link(cluster, "network:features")
    return net
