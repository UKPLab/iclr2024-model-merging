__all__ = ["acoustic_model_config"]

import i6_core.rasr as rasr


def acoustic_model_config(
    state_tying="monophone",
    states_per_phone=3,
    state_repetitions=1,
    across_word_model=True,
    early_recombination=False,
    tdp_scale=1.0,
    tdp_transition=(3.0, 0.0, 3.0, 2.0),
    tdp_silence=(0.0, 3.0, "infinity", 6.0),
    tying_type="global",
    nonword_phones="",
    tdp_nonword=(0.0, 3.0, "infinity", 6.0),
):
    config = rasr.RasrConfig()

    config.state_tying.type = state_tying
    config.allophones.add_from_lexicon = True
    config.allophones.add_all = False

    config.hmm.states_per_phone = states_per_phone
    config.hmm.state_repetitions = state_repetitions
    config.hmm.across_word_model = across_word_model
    config.hmm.early_recombination = early_recombination

    config.tdp.scale = tdp_scale

    config.tdp["*"].loop = tdp_transition[0]
    config.tdp["*"].forward = tdp_transition[1]
    config.tdp["*"].skip = tdp_transition[2]
    config.tdp["*"].exit = tdp_transition[3]

    config.tdp.silence.loop = tdp_silence[0]
    config.tdp.silence.forward = tdp_silence[1]
    config.tdp.silence.skip = tdp_silence[2]
    config.tdp.silence.exit = tdp_silence[3]

    config.tdp["entry-m1"].loop = "infinity"
    config.tdp["entry-m2"].loop = "infinity"

    if tying_type == "global-and-nonword":
        config.tdp.tying_type = "global-and-nonword"
        config.tdp.nonword_phones = nonword_phones
        for nw in [0, 1]:
            k = "nonword-%d" % nw
            config.tdp[k].loop = tdp_nonword[0]
            config.tdp[k].forward = tdp_nonword[1]
            config.tdp[k].skip = tdp_nonword[2]
            config.tdp[k].exit = tdp_nonword[3]

    return config
