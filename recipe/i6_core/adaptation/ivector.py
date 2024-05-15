__all__ = ["IVectorTrainingJob", "IVectorExtractionJob"]

import sys
import logging
import numpy as np
import tempfile
import shutil

from sisyphus import *

Path = setup_path(__package__)

try:
    from bob.learn.em import GMMMachine, GMMStats, IVectorMachine, IVectorTrainer
    from bob.io.base import HDF5File
except ImportError:
    raise ImportError(
        'No module named "bob".\nInstallation guide: https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/doc/install.html\nAlternatively use this python: /work/asr3/kitza/software/miniconda3/envs/bob_py3/bin/python'
    )

import i6_core.rasr as rasr
import i6_core.util as util
import i6_core.lib.rasr_cache as sc

"""
If filesystem locking problems occur set:
    DEFAULT_ENVIRONMENT_SET['HDF5_USE_FILE_LOCKING']='FALSE'

tested with:
    bob                       4.0.1
    bob.io.base               3.0.3
    bob.learn.em              2.1.1


1. train ubm mixtures with starting at sdm:
    
    feature_flow = self.feature_flows[corpus][feature_flow]   
    segment_job = corpus_recipes.SegmentCorpus(self.crp[corpus].corpus_config.file, 1)
    map_job = ubm.UbmWarpingMapJob(segment_list=segment_job.single_segment_files[1])
    mono_job = ubm.EstimateWarpingMixturesJob(crp=self.crp[corpus],
                                              old_mixtures=None,
                                              feature_flow=feature_flow,
                                              warping_map=map_job.warping_map,
                                              warping_factors=map_job.alphas_file,
                                              split_first=False, )
    
    seq = ubm.TrainWarpingFactorsSequence(crp=self.crp[corpus],
                                          initial_mixtures=mono_job.mixtures,
                                          feature_flow=feature_flow,
                                          warping_map=map_job.warping_map,
                                          warping_factors=map_job.alphas_file,
                                          action_sequence=['accumulate'] + meta.split_and_accumulate_sequence(splits, accs_per_split))

2. Generate feature and alignment caches for each speaker/cluster


3. Train I-Vector Model

    ivec_train_args = {
      'crp'                 : system.crp['train_per_speaker'],
      'ubm'                 : system.mixtures['train']['ubm'][-1],
      'features'            : system.feature_caches['train_per_speaker']['mfcc'].hidden_paths,
      'alignment'           : system.alignments['train_per_speaker']['tri_1'].alternatives['task_dependent'].hidden_paths,
      'allophones'          : system.allophone_files['base'],
      'dim'                 : 100,
      'allophones_to_ignore': ['laughs', 'noise', 'sil', 'inaudible', 'spn'],
      }
    ivec_train_job = IVectorTrainingJob(**ivec_train_args)


4. Extract I-Vectors

    ivec_extract_args = {
      'crp'                 : system.crp['train_per_speaker'],
      'ubm'                 : system.mixtures['train']['ubm'][5],
      'features'            : system.feature_caches['train_per_speaker']['mfcc'].hidden_paths,
      'alignment'           : system.alignments['train_per_speaker']['tri_1'].alternatives['task_dependent'].hidden_paths,
      'allophones'          : system.allophone_files['base'],
      'dim'                 : 100,
      'allophones_to_ignore': ['laughs', 'noise', 'sil', 'inaudible', 'spn'],
      't_matrix'            : ivec_train_job.t_matrix,
      }
    j = IVectorExtractionJob(**ivec_extract_args)
"""


def convert_gmm(gmm):
    """
    Converts rasr_cache.MixtureFile to bob.em.learn.GMMachine
    :param gmm: (MixtureFile)
    :return: (GMMachine)
    """
    ubm = GMMMachine(gmm.nMeans, gmm.dim)

    tmp_m = np.ndarray((gmm.nMeans, gmm.dim))
    tmp_c = np.ndarray((gmm.nCovs, gmm.dim))
    for i in range(gmm.nMeans):
        tmp_m[i, :] = np.array(gmm.getMeanByIdx(i))
        tmp_c[i, :] = np.array(
            gmm.getCovByIdx(0)
        )  # TODO figure out to generate same number of covariances as means
    ubm.means = tmp_m
    ubm.variances = tmp_c
    return ubm


def concat_features_with_ivec(feature_net, ivec_path):
    """
    Generate a new flow-network with i-vectors repeated and concatenated to original feature stream
    :param feature_net: original flow-network
    :param ivec_path: ivec_path from IVectorExtractionJob
    :return:
    """
    # copy original net
    net = rasr.FlowNetwork(name=feature_net.name)
    net.add_param(["id", "start-time", "end-time"])
    net.add_output("features")
    mapping = net.add_net(feature_net)
    net.interconnect_inputs(feature_net, mapping)

    # load ivec cache and repeat
    fc = net.add_node(
        "generic-cache", "feature-cache-ivec", {"id": "$(id)", "path": ivec_path}
    )
    sync = net.add_node("signal-repeating-frame-prediction", "sync")
    net.link(fc, sync)
    for node in feature_net.get_output_links("features"):
        net.link(node, "%s:%s" % (sync, "target"))

    # concat original feature output with repeated ivecs
    concat = net.add_node(
        "generic-vector-f32-concat",
        "concatenation",
        {"check-same-length": True, "timestamp-port": "feature-1"},
    )
    for node in feature_net.get_output_links("features"):
        net.link(node, "%s:%s" % (concat, "feature-1"))
    net.link(sync, "%s:%s" % (concat, "feature-2"))

    net.link(concat, "network:features")

    return net


class IVectorTrainingJob(Job):
    """
    Trains a IVectorMachine using the python lib bob
    """

    def __init__(
        self,
        crp,
        ubm,
        features,
        alignment,
        allophones,
        dim,
        allophones_to_ignore,
        iter=10,
        rqmt=None,
    ):
        """
        :param crp: (CommonRasrParameters) need for concurrency
        :param ubm: (Path) to UBM trained with ubm.TrainWarpingFactorsSequence
        :param features: system.feature_caches['corpus']['mfcc'].hidden_paths; gone feature.cache file per i-vector, good features are mfcc, plp
        :param alignment: system.alignments['corpus'][''].alternatives['task_dependent'].hidden_paths; one alignment.cache file per i-vector
        :param allophones: system.allophone_files['base']
        :param dim: (int) dimension of the i-vec, usually between 50-400
        :param allophones_to_ignore: list(string) ['laughs', 'noise', 'sil', 'inaudible', 'spn']
        :param iter: (int) number of em iterations during ivector training
        :param rqmt:
        """
        self.crp = crp
        self.ubm = ubm
        self.features = features
        self.alignment = alignment
        self.allophones = allophones
        self.dim = dim
        self.allophones_to_ignore = allophones_to_ignore
        self.iter = iter

        self.concurrent = crp.concurrent
        self.rqmt = rqmt if rqmt else {"time": 1, "cpu": 1, "gpu": 0, "mem": 1}

        self.single_accu_caches = dict(
            (i, self.output_path("accu.%d" % i, cached=True))
            for i in range(1, self.concurrent + 1)
        )
        self.accu_path = util.MultiOutputPath(
            self, "accu.$(TASK)", self.single_accu_caches, cached=True
        )
        self.t_matrix = self.output_path("t.matrix")

    def tasks(self):
        yield Task("acc", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("est", rqmt=self.rqmt)

    def acc(self, task_id):
        mix_file = util.cache_path(self.ubm)
        align_file = util.cache_path(self.alignment[task_id])
        feat_file = util.cache_path(self.features[task_id])
        allo_file = util.cache_path(self.allophones)

        logging.info("Reading mixture file from '%s'..." % mix_file)
        gmm = sc.MixtureSet(mix_file)
        logging.info(
            "Read %d means and %d covariances of dimension %d"
            % (gmm.nMeans, gmm.nCovs, gmm.dim)
        )

        ubm = convert_gmm(gmm)

        ivm = IVectorMachine(ubm, self.dim)
        ivm.variance_threshold = 1e-5

        gs = GMMStats(gmm.nMeans, gmm.dim)

        logging.info(
            "Opening alignment cache '%s' with allophones from '%s'; ignoring '%s'"
            % (align_file, allo_file, ",".join(self.allophones_to_ignore))
        )
        aligncache = sc.FileArchive(align_file)
        aligncache.setAllophones(allo_file)

        cache = sc.FileArchive(feat_file)

        for a in cache.ft.keys():
            if a.endswith(".attribs"):
                continue
            logging.info("Reading '%s'..." % a)

            time, data = cache.read(a, "feat")

            align = aligncache.read(a, "align")
            if len(align) < 1:
                logging.warning("No data for segment: '%s' in alignment." % a)
                continue
            allos = []
            for (t, i, s, w) in align:
                allos.append(aligncache.allophones[i])
            allos = list(aligncache.allophones[i] for (t, i, s, w) in align)
            T = len(list(filter(lambda al: al not in self.allophones_to_ignore, allos)))

            feat = np.ndarray((T, len(data[0])))
            k = 0

            for t in range(len(data)):
                (_, allo, state, weight) = align[t]
                if aligncache.allophones[allo] not in self.allophones_to_ignore:
                    feat[k, :] = data[t]
                    k += 1

            ivm.ubm.acc_statistics(feat, gs)

        logging.info(
            "Writing Gaussian statistics to '%s'"
            % self.single_accu_caches[task_id].get_path()
        )
        gs.save(HDF5File(self.single_accu_caches[task_id].get_path(), "w"))

    def est(self):
        mix_file = util.cache_path(self.ubm)
        ivecdim = self.dim

        gslist = []
        for idx, gfile in self.single_accu_caches.items():
            gs = GMMStats(HDF5File(tk.uncached_path(gfile)))
            gslist.append(gs)

        gmm = sc.MixtureSet(mix_file)
        ubm = convert_gmm(gmm)

        ivm = IVectorMachine(ubm, ivecdim)
        ivm.variance_threshold = 1e-5

        ivtrainer = IVectorTrainer(update_sigma=True)
        ivtrainer.initialize(ivm, gslist)

        for i in range(self.iter):
            ivtrainer.e_step(ivm, gslist)
            ivtrainer.m_step(ivm)

        ivm.save(HDF5File(self.t_matrix.get_path(), "w"))


class IVectorExtractionJob(Job):
    """
    Does extraction of i-vectors given a model from IVectorTrainingJob
    """

    def __init__(
        self,
        crp,
        t_matrix,
        ubm,
        features,
        alignment,
        allophones,
        dim,
        allophones_to_ignore,
        length_norm=True,
        rqmt=None,
    ):
        """
        :param crp: (CommonRasrParameters) need for concurrency
        :param t_matrix: (HDF5File) IVectorTrainingJob.t_matrix, contains learned ubm and JFA
        :param ubm: (Path) to UBM trained with ubm.TrainWarpingFactorsSequence
        :param features: system.feature_caches['corpus']['mfcc'].hidden_paths; gone feature.cache file per i-vector, good features are mfcc, plp
        :param alignment: system.alignments['corpus'][''].alternatives['task_dependent'].hidden_paths; one alignment.cache file per i-vector
        :param allophones: system.allophone_files['base']
        :param dim: (int) dimension of the i-vec, usually between 50-400
        :param allophones_to_ignore: list(string) ['laughs', 'noise', 'sil', 'inaudible', 'spn']
        :param length_norm: (bool) normalize i-vector to unit length
        :param rqmt:
        """
        self.crp = crp
        self.ubm = ubm
        self.t_matrix = t_matrix
        self.features = features
        self.alignment = alignment
        self.allophones = allophones
        self.dim = dim
        self.allophones_to_ignore = allophones_to_ignore
        self.length_norm = length_norm

        self.concurrent = crp.concurrent
        self.rqmt = rqmt if rqmt else {"time": 1, "cpu": 1, "gpu": 0, "mem": 1}

        self.single_ivec_caches = dict(
            (i, self.output_path("ivec.%d" % i, cached=True))
            for i in range(1, self.concurrent + 1)
        )
        self.ivec_path = util.MultiOutputPath(
            self, "ivec.$(TASK)", self.single_ivec_caches, cached=True
        )

    def tasks(self):
        yield Task("forward", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

    def forward(self, task_id):
        mixfile = util.cache_path(self.ubm)
        ivmfile = tk.uncached_path(self.t_matrix)
        alignfile = util.cache_path(self.alignment[task_id])
        allofile = tk.uncached_path(self.allophones)
        alloignore = self.allophones_to_ignore
        featfile = util.cache_path(self.features[task_id])
        ivecdim = self.dim
        lengthnorm = bool(self.length_norm)

        gmm = sc.MixtureSet(mixfile)
        ubm = convert_gmm(gmm)

        ivm = IVectorMachine(ubm, ivecdim)
        ivm.load(HDF5File(ivmfile))

        tmp_ivec_file = tempfile.mktemp(suffix=".ivec")

        out = sc.FileArchive(tmp_ivec_file)

        logging.info(
            "Opening alignment cache '%s' with allophones from '%s'; ignoring '%s'"
            % (alignfile, allofile, ",".join(alloignore))
        )
        aligncache = sc.FileArchive(alignfile)
        aligncache.setAllophones(allofile)

        cache = sc.FileArchive(featfile)
        cur_rec = ""
        tmp_feat = None
        tmp_segs = []

        for a in sorted(cache.ft.keys()):
            if a.endswith(".attribs"):
                continue
            logging.info("Reading '%s'..." % a)
            ncorpus, nrec, nseg = a.split("/")

            try:
                time, data = cache.read(a, "feat")

                align = aligncache.read(a, "align")
                allos = list(aligncache.allophones[i] for (t, i, s, w) in align)
                T = len(list(filter(lambda al: al not in alloignore, allos)))

                feat = np.ndarray((T, len(data[0])))
                k = 0
                for t in range(len(data)):
                    (_, allo, state, weight) = align[t]
                    if aligncache.allophones[allo] not in alloignore:
                        feat[k, :] = data[t]
                        k += 1

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logging.error("failed", sys.exc_info(), exc_tb.tb_lineno)
                ivector = np.zeros([1, ivecdim])
                out.addFeatureCache(a, [ivector], [[0.0, 999999.0]])
                continue

            if nrec == cur_rec:
                tmp_feat = np.concatenate((tmp_feat, feat), axis=0)
                tmp_segs.append(a)
                continue
            else:
                if cur_rec != "":
                    gs_test = GMMStats(gmm.nMeans, gmm.dim)
                    ivm.ubm.acc_statistics(tmp_feat, gs_test)
                    ivector = ivm.project(gs_test)
                    ivector = ivector / np.linalg.norm(ivector)
                    ivector = np.expand_dims(ivector, 0)
                    for seg in tmp_segs:
                        out.addFeatureCache(seg, [ivector], [[0.0, 999999.0]])

                tmp_feat = feat
                tmp_segs = [a]
                cur_rec = nrec

        # last rec
        gs_test = GMMStats(gmm.nMeans, gmm.dim)
        ivm.ubm.acc_statistics(tmp_feat, gs_test)
        ivector = ivm.project(gs_test)
        if lengthnorm:
            ivector = ivector / np.linalg.norm(ivector)
        ivector = np.expand_dims(ivector, 0)
        for seg in tmp_segs:
            out.addFeatureCache(seg, [ivector], [[0.0, 999999.0]])

        out.finalize()

        del out  # delete this to close the file handle. This ensures all data is written.

        shutil.move(tmp_ivec_file, self.single_ivec_caches[task_id].get_path())
