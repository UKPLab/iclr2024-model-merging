__all__ = ["add_context_flow"]

import i6_core.rasr as rasr


def add_context_flow(
    feature_net,
    max_size=9,
    right=4,
    margin_condition="present-not-empty",
    expand_timestamp=False,
):
    net = rasr.FlowNetwork()
    net.add_output("features")

    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    context = net.add_node(
        "signal-vector-f32-sequence-concatenation",
        "context-window",
        {
            "max-size": max_size,
            "right": right,
            "margin-condition": margin_condition,
            "expand-timestamp": expand_timestamp,
        },
    )
    net.link(mapping[feature_net.get_output_links("features").pop()], context)
    net.link(context, "network:features")

    return net
