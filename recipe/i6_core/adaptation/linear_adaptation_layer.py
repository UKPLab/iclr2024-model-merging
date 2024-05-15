__all__ = [
    "MetaLinearAdaptationJob",
    "WriteFileJob",
    "MergeFilesJob",
]

import copy
import os

from sisyphus import *

Path = setup_path(__package__)

import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.corpus as corpus_recipes
import i6_core.util as util
import i6_core.recognition as recog
from i6_core.util import MultiOutputPath

# we do not put features in caches or calc alignment here, that is part of the setup
# nothing should be hard coded here, only modules


class MetaLinearAdaptationJob(Job):
    def __init__(
        self,
        name,
        corpus,
        num_clusters,
        segment_dir,
        original_config,
        original_model,
        returnn_training_args,  # should contain feature_flow, alignment, num_classes , ...
        returnn_scorer_args,
        returnn_recog_args,
        wer_scorer,
        wer_scorer_args,
        input_trans_args=None,
        hidden_trans_args=None,
        output_trans_args=None,
    ):
        self.name = name
        self.corpus = corpus
        self.segment_dir = segment_dir
        self.num_clusters = num_clusters
        self.original_config = original_config
        self.original_model = original_model

        self.returnn_training_args = returnn_training_args
        self.scorer_args = returnn_scorer_args
        self.recog_args = returnn_recog_args
        self.wer_scorer = wer_scorer
        self.wer_scorer_args = wer_scorer_args

        self.input_trans_args = input_trans_args
        self.hidden_trans_args = hidden_trans_args
        self.output_trans_args = output_trans_args

        self.recognition_keep_value = 5
        self.adapt_config = None
        self.single_segments = None
        self.job_created = False
        self.jobs = {}
        self.ctm_files = {}
        self.scorers = {}
        save_interval = returnn_training_args["save_interval"]
        num_epochs = returnn_training_args["num_epochs"]
        self.epochs = list(range(save_interval, num_epochs, save_interval)) + [
            num_epochs
        ]

        # Outputs
        self.lattice_dir = self.output_path("lattice", True)

        self.models = {}
        self.model_dir = self.output_path("models", True)
        self.model_path = MultiOutputPath(
            self, "models/cluster.$(CLUSTER)", self.model_dir
        )

        self.best_wer = 1000
        self.best_model = "none"
        self.wer = self.output_path("wer.txt")
        self.config_dir = self.output_path("returnn_recog_config", True)
        self.recog_config = self.output_path("returnn_recog.config")
        # self.recog_config = { i : self.output_path('returnn_config.%d' % i) for i in list(range(save_interval, num_epochs, save_interval)) + [num_epochs] }

    def tasks(self):
        # num_cluster = util.get_val(self.num_clusters)
        yield Task("run", mini_task=True)
        # yield Task('recognition', args=range(1,num_cluster+1))
        # yield Task('combine_models', mini_task=True) # combine best epochs to one model

    def update(self):
        if not self.job_created:
            self.collect_files()

            self.modify_config()

            self.setup_training()

            self.lattice_combined_recognition()
            # self.recognition()

    def collect_files(self):
        self.single_segments = {
            i: self.segment_dir.get_path() + ("/speaker.%d" % i)
            for i in range(1, util.get_val(self.num_clusters) + 1)
        }

    def modify_config(self):
        self.adapt_config = self.original_config
        network = self.original_config["network"]

        for name, layer in network.items():
            layer["trainable"] = False

        if self.input_trans_args:
            self.insert_input_layer()
        if self.hidden_trans_args:
            self.insert_hidden_layer(
                network, self.original_model.model, self.hidden_trans_args
            )
        if self.output_trans_args:
            self.insert_output_layer()

        # self.adapt_config['save_func'] = ['partial']

    def insert_input_layer(self):
        pass

    def insert_hidden_layer(self, network, model_file, insert_layer_args):
        network_load_layers = {}
        for key in network:
            network_load_layers[key] = key
        network_save_layers = []
        for key in insert_layer_args:
            network_save_layers.append(key)

        for trans_layer in insert_layer_args:
            target_layer = insert_layer_args[trans_layer]["preceding_layer"]
            layer = copy.deepcopy(insert_layer_args[trans_layer]["layer"])
            layer["from"] = [target_layer]
            if target_layer in network:  # check if layer to transform is in network
                # search all folling layers and replace preceding layer with target layer
                for following_layer in network:
                    if target_layer in network[following_layer]["from"]:
                        index = network[following_layer]["from"].index(target_layer)
                        network[following_layer]["from"][index] = trans_layer

                    n_out = network[target_layer]["n_out"]
                    layer["n_out"] = n_out
                    # layer['forward_weights_init'] = 'Identity' #'numpy.eye({})'.format(layer['n_out'])
                network[trans_layer] = layer
        # self.adapt_config.update(network)
        self.adapt_config["import_merge_models"] = {model_file: network_load_layers}
        self.adapt_config["save_layer"] = network_save_layers

    def insert_output_layer(self):
        pass

    def run(self):
        for key in self.single_segments:
            j = self.jobs["train_{}".format(key)]
            os.symlink(
                os.path.abspath(j.model_dir.get_path()),
                self.model_dir.get_path() + "/cluster.%s" % key,
            )

        with open(self.wer.get_path(), "w") as wer_file:
            for name, s in self.scorers.items():
                wer = s.calc_wer()
                wer_file.write("{} : {}\n".format(name, wer))
                if wer < self.best_wer:
                    self.best_wer = wer
                    self.best_wer_name = name

    def setup_training(self):
        for key, seg in self.single_segments.items():
            print("{}: Adding training {}".format(self.name, key))
            new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(seg)
            seg_train = new_segments.out_segments["train"]
            seg_dev = new_segments.out_segments["dev"]

            train_corpus = copy.deepcopy(self.corpus)
            train_corpus.out_segment_path = seg_train
            train_corpus.concurrent = 1

            dev_corpus = copy.deepcopy(self.corpus)
            dev_corpus.out_segment_path = seg_dev
            dev_corpus.concurrent = 1

            j = returnn.ReturnnRasrTrainingJob(
                train_crp=train_corpus,
                dev_crp=dev_corpus,
                returnn_config=returnn.ReturnnConfig(self.adapt_config),
                **self.returnn_training_args,
            )
            self.jobs["train_{}".format(key)] = j
            self.models[key] = j.out_models
            self.add_input(j.out_plot_lr)

            for key, value in j.out_models.items():
                self.add_input(value.model)
            # self.add_input(j.models)
            # for m in j.models:
            #    self.add_input(j.models[m].model)
            self.job_created = True

    def make_recog_config(self, job, file_path, import_models):
        import json
        import pprint
        import string
        import textwrap

        config = job.returnn_config
        config.update(job.returnn_post_config)
        config.update(import_models)
        config["num_outputs"]["classes"] = [util.get_val(job.num_classes), 1]

        config_lines = []
        unreadable_data = {}

        pp = pprint.PrettyPrinter(indent=2, width=150)
        for k, v in sorted(config.items()):
            if pprint.isreadable(v):
                config_lines.append("%s = %s" % (k, pp.pformat(v)))
            else:
                unreadable_data[k] = v

        if len(unreadable_data) > 0:
            config_lines.append("import json")
            print(unreadable_data)
            json_data = json.dumps(unreadable_data).replace('"', '\\"')
            config_lines.append('config = json.loads("%s")' % json_data)
        else:
            config_lines.append("config = {}")

        python_code = string.Template(job.PYTHON_CODE).substitute(
            {
                "PROLOG": job.returnn_config.python_prolog,
                "REGULAR_CONFIG": "\n".join(config_lines),
                "EPILOG": job.returnn_config.python_epilog,
            }
        )
        # with open(file_path, 'wt', encoding='utf-8') as f:
        #    f.write(python_code)
        return python_code

    def lattice_combined_recognition(self):
        for epoch in self.epochs:
            lattice_bundles = []
            epoch_name = "{}_epoch.{}".format(self.name, epoch)
            print("{}: Adding recognition for epoch {}".format(self.name, epoch_name))
            for key, segment in self.single_segments.items():
                eval_corpus = copy.deepcopy(self.corpus)
                eval_corpus.out_segment_path = segment
                eval_corpus.concurrent = 1

                bundled_flow = self.recog_args["feature_flow"]
                bundled_flow.flags["cache_mode"] = "bundle"
                scorer_name = "{}_{}".format(epoch_name, key)

                returnn_scorer = rasr.ReturnnScorer(
                    feature_dimension=self.scorer_args["feature_dimension"],
                    output_dimension=self.scorer_args["output_dimension"],
                    prior_mixtures=self.scorer_args["prior_mixtures"],
                    model=self.models[key][epoch],
                    prior_scale=self.scorer_args["prior_scale"],
                    prior_file=None,
                )

                eval_corpus.language_model_config.scale = self.recog_args["lm_scale"]
                model_combination_config = rasr.RasrConfig()
                model_combination_config.pronunciation_scale = self.recog_args[
                    "pronunciation_scale"
                ]

                rec = recog.AdvancedTreeSearchJob(
                    crp=eval_corpus,
                    feature_flow=bundled_flow,
                    feature_scorer=returnn_scorer,
                    model_combination_config=model_combination_config,
                )
                rec.keep_value(self.recognition_keep_value)
                rec.set_vis_name("Recog %s" % scorer_name)
                self.jobs["recog_%s" % scorer_name] = rec
                lattice_bundles.append(rec.out_lattice_bundle)
            m = MergeFilesJob(lattice_bundles)

            self.jobs["lat2ctm_%s" % epoch_name] = lat2ctm = recog.LatticeToCtmJob(
                crp=self.corpus, lattice_cache=m.out_file, parallelize=False
            )
            self.ctm_files["recog_%s" % epoch_name] = lat2ctm.out_ctm_file

            kwargs = copy.deepcopy(self.wer_scorer_args)
            kwargs["hyp"] = lat2ctm.out_ctm_file
            scorer = self.wer_scorer(**kwargs)

            self.jobs["scorer_%s" % epoch_name] = scorer
            self.scorers[epoch_name] = scorer
            tk.register_output("recog_%s.reports" % epoch_name, scorer.report_dir)
            self.add_input(scorer.report_dir)


class WriteFileJob(Job):
    def __init__(self, input):
        self.input = input
        self.file = self.output_path("file")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.file.get_path(), "w") as out_file:
            out_file.writelines(self.input)


class MergeFilesJob(Job):
    def __init__(self, input):
        self.input = input
        self.out_file = self.output_path("files.bundle", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.out_file.get_path(), "w") as out_file:
            for f in self.input:
                in_file = open(f.get_path(), "r")
                out_file.writelines(in_file)
