__all__ = ["VTLNFeaturesJob"]

import i6_core.features as features
from .flow import warp_filterbank_with_map_flow


def VTLNFeaturesJob(
    crp,
    feature_flow,
    map_file,
    extra_warp_args=None,
    extra_config=None,
    extra_post_config=None,
):
    if extra_warp_args is None:
        extra_warp_args = {}

    feature_flow = warp_filterbank_with_map_flow(
        feature_flow, map_file, **extra_warp_args
    )

    return features.FeatureExtractionJob(
        crp=crp,
        feature_flow=feature_flow,
        port_name_mapping={"features": "vtln"},
        job_name="VTLN",
        rtf=0.1,
        mem=2,
        extra_config=extra_config,
        extra_post_config=extra_post_config,
    )
