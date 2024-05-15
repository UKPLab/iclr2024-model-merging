__all__ = [
    "BasicCartQuestions",
    "CMUCartQuestions",
    "BeepCartQuestions",
    "PythonCartQuestions",
]

import collections
import copy
import itertools as it
import xml.etree.ElementTree as ET

from sisyphus import *

from i6_core.util import write_xml

Path = setup_path(__package__)


class BasicCartQuestions:
    def __init__(self, phoneme_path, max_leaves, min_obs):
        self.phoneme_path = phoneme_path
        self.max_leaves = max_leaves
        self.min_obs = min_obs

    def load_phonemes_from_file(self):
        with open(tk.uncached_path(self.phoneme_path), "r") as phoneme_file:
            phonemes = ["#"] + [
                l.strip().lower().split("\t")[0] for l in phoneme_file.readlines()
            ]
            if "sil" not in phonemes:
                phonemes.append("sil")
        return phonemes

    def get_questions(self):
        phonemes = self.load_phonemes_from_file()

        root = ET.Element("decision-tree-training")
        property_def = ET.SubElement(root, "properties-definition")
        properties = collections.OrderedDict(
            [
                ("hmm-state", ["0", "1", "2"]),
                (
                    "boundary",
                    [
                        "within-lemma",
                        "begin-of-lemma",
                        "end-of-lemma",
                        "single-phoneme-lemma",
                    ],
                ),
                ("history[0]", phonemes),
                ("central", phonemes),
                ("future[0]", phonemes),
            ]
        )

        for prop, values in properties.items():
            ET.SubElement(property_def, "key").text = prop
            value_map = ET.SubElement(property_def, "value-map")
            for id, value_text in enumerate(values):
                ET.SubElement(value_map, "value", id=str(id)).text = value_text

        ET.SubElement(root, "max-leaves").text = str(self.max_leaves)

        step = ET.SubElement(root, "step", name="silence", action="cluster")
        question = ET.SubElement(ET.SubElement(step, "questions"), "question")
        ET.SubElement(question, "key").text = "central"
        ET.SubElement(question, "value").text = "sil"

        step = ET.SubElement(root, "step", name="central", action="partition")
        ET.SubElement(step, "min-obs").text = str(self.min_obs)
        question = ET.SubElement(
            ET.SubElement(ET.SubElement(step, "questions"), "for-each-value"),
            "question",
            description="central",
        )
        ET.SubElement(question, "key").text = "central"

        step = ET.SubElement(root, "step", name="hmm-state", action="partition")
        ET.SubElement(step, "min-obs").text = str(self.min_obs)
        question = ET.SubElement(
            ET.SubElement(ET.SubElement(step, "questions"), "for-each-value"),
            "question",
            description="hmm-state",
        )
        ET.SubElement(question, "key").text = "hmm-state"

        step = ET.SubElement(root, "step", name="linguistics", action="partition")
        ET.SubElement(step, "min-obs").text = str(self.min_obs)
        questions = ET.SubElement(step, "questions")

        question = ET.SubElement(
            ET.SubElement(questions, "for-each-value"),
            "question",
            description="boundary",
        )
        ET.SubElement(question, "key").text = "boundary"

        return root

    def write_to_file(self, file):
        write_xml(file, self.get_questions())


class CMUCartQuestions(BasicCartQuestions):
    def __init__(self, include_central_phoneme=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_central_phoneme = include_central_phoneme

    def get_questions(self):
        root = super().get_questions()

        with open(tk.uncached_path(self.phoneme_path), "r") as phoneme_file:
            phonemes = it.groupby(
                sorted(
                    [l.strip().lower().split("\t") for l in phoneme_file.readlines()],
                    key=lambda l: l[1],
                ),
                key=lambda l: l[1],
            )

        questions = root.find('./step[@name="linguistics"]/questions')
        if questions is None:
            raise KeyError("could not find linguistics step")
        k = "history[0] %sfuture[0]" % (
            "central " if self.include_central_phoneme else ""
        )
        fekey = ET.SubElement(questions, "for-each-key", keys=k)
        ET.SubElement(
            ET.SubElement(fekey, "for-each-value"),
            "question",
            description="context-phone",
        )
        for k, g in phonemes:
            ET.SubElement(
                ET.SubElement(fekey, "question", description=k), "values"
            ).text = " ".join(e[0] for e in g)

        return root

    def __sis_state__(self):
        state = copy.copy(self.__dict__)
        if not self.include_central_phoneme:
            del state["include_central_phoneme"]
        return state


class BeepCartQuestions(BasicCartQuestions):
    def __init__(self, include_central_phoneme=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_central_phoneme = include_central_phoneme

    def get_questions(self):
        root = super().get_questions()

        phonemes = [  # classes based on cmu dict
            (
                "vowel",
                [
                    "aa",
                    "ae",
                    "ah",
                    "ao",
                    "aw",
                    "ax",
                    "ay",
                    "ea",
                    "eh",
                    "er",
                    "ey",
                    "ia",
                    "ih",
                    "iy",
                    "oh",
                    "ow",
                    "oy",
                    "ua",
                    "uh",
                    "uw",
                ],
            ),
            ("stop", ["b", "d", "g", "k", "p", "t"]),
            ("affricate", ["ch", "jh"]),
            ("fricative", ["dh", "f", "s", "sh", "th", "v", "z", "zh"]),
            ("aspirate", ["hh"]),
            ("liquid", ["l", "r"]),
            ("nasal", ["m", "n", "ng"]),
            ("semivowel", ["w", "y"]),
            # vowels
            ("vowel_a", ["aa", "ae", "ah", "ao", "aw", "ax", "ay"]),
            ("vowel_e", ["ea", "eh", "er", "ey"]),
            ("vowel_i", ["ia", "ih", "iy"]),
            ("vowel_o", ["oh", "ow", "oy"]),
            ("vowel_u", ["ua", "uh", "uw"]),
        ]

        questions = root.find('./step[@name="linguistics"]/questions')
        if questions is None:
            raise KeyError("could not find linguistics step")
        k = "history[0] %sfuture[0]" % (
            "central " if self.include_central_phoneme else ""
        )
        fekey = ET.SubElement(questions, "for-each-key", keys=k)
        ET.SubElement(
            ET.SubElement(fekey, "for-each-value"),
            "question",
            description="context-phone",
        )
        for k, g in phonemes:
            ET.SubElement(
                ET.SubElement(fekey, "question", description=k), "values"
            ).text = " ".join(g)

        return root


class PythonCartQuestions:
    def __init__(self, phonemes, steps, max_leaves=9001, hmm_states=3):
        self.phonemes = phonemes
        self.steps = steps
        self.max_leaves = max_leaves
        self.hmm_states = hmm_states

    def get_questions(self):
        root = ET.Element("decision-tree-training")
        property_def = ET.SubElement(root, "properties-definition")
        properties = [
            (
                "boundary",
                [
                    "within-lemma",
                    "begin-of-lemma",
                    "end-of-lemma",
                    "single-phoneme-lemma",
                ],
            ),
            ("history[0]", self.phonemes),
            ("central", self.phonemes),
            ("future[0]", self.phonemes),
        ]
        if self.hmm_states > 1:
            properties = [
                ("hmm-state", [str(i) for i in range(self.hmm_states)])
            ] + properties
        properties = collections.OrderedDict(properties)

        for prop, values in properties.items():
            ET.SubElement(property_def, "key").text = prop
            value_map = ET.SubElement(property_def, "value-map")
            for id, value_text in enumerate(values):
                ET.SubElement(value_map, "value", id=str(id)).text = value_text

        ET.SubElement(root, "max-leaves").text = str(self.max_leaves)

        def process_questions(root, questions):
            for q in questions:
                if q["type"] == "for-each-value":
                    process_questions(
                        ET.SubElement(root, "for-each-value"), q["questions"]
                    )
                elif q["type"] == "for-each-key":
                    process_questions(
                        ET.SubElement(root, "for-each-key", keys=q["keys"]),
                        q["questions"],
                    )
                elif q["type"] == "question":
                    e = ET.SubElement(root, "question")
                    desc = q.get("description", None)
                    if desc is not None:
                        e.set("description", desc)
                    for k in ["key", "value", "values"]:
                        v = q.get(k, None)
                        if v is not None:
                            ET.SubElement(e, k).text = v

        for s in self.steps:
            step = ET.SubElement(root, "step", name=s["name"])
            action = s.get("action", None)
            if action is not None:
                step.set("action", s["action"])
            if "min-obs" in s:
                ET.SubElement(step, "min-obs").text = str(s["min-obs"])

            questions = ET.SubElement(step, "questions")
            process_questions(questions, s["questions"])

        return root

    def write_to_file(self, file):
        write_xml(file, self.get_questions())
