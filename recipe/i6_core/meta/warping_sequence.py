__all__ = ["TrainWarpingFactorsSequence"]

import i6_core.vtln as vtln


class TrainWarpingFactorsSequence:
    def __init__(
        self,
        crp,
        initial_mixtures,
        feature_flow,
        warping_map,
        warping_factors,
        action_sequence,
        split_extra_args=None,
        accumulate_extra_args=None,
        seq_extra_args=None,
    ):
        split_extra_args = {} if split_extra_args is None else split_extra_args
        accumulate_extra_args = (
            {} if accumulate_extra_args is None else accumulate_extra_args
        )
        seq_extra_args = {} if seq_extra_args is None else seq_extra_args

        self.action_sequence = action_sequence

        self.all_jobs = []
        self.all_logs = []
        self.all_mixtures = []
        self.selected_mixtures = []

        current_mixtures = initial_mixtures

        for idx, action in enumerate(action_sequence):
            split = action.startswith("split")
            args = {
                "crp": crp,
                "old_mixtures": current_mixtures,
                "feature_flow": feature_flow,
                "warping_map": warping_map,
                "warping_factors": warping_factors,
                "split_first": split,
            }
            args.update(split_extra_args if split else accumulate_extra_args)
            if idx in seq_extra_args:
                args.update(seq_extra_args[idx])

            j = vtln.EstimateWarpingMixturesJob(**args)
            self.all_jobs.append(j)
            self.all_logs.append(j.log_file)
            self.all_mixtures.append(j.out_mixtures)

            current_mixtures = j.out_mixtures

            if action[-1] == "!":
                self.selected_mixtures.append(j.out_mixtures)
