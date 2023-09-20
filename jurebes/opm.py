from ovos_classifiers.tasks.tagger import OVOSNgramTagger
from ovos_plugin_manager.templates.pipeline import IntentPipelinePlugin, IntentMatch
from ovos_utils import classproperty
from ovos_utils.log import LOG
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from jurebes import JurebesIntentContainer


def _munge(name, skill_id):
    return f"{name}:{skill_id}"


def _unmunge(munged):
    return munged.split(":", 2)


class JurebesPipelinePlugin(IntentPipelinePlugin):
    id2clf = {  # mappings from mycroft.conf strings to classifier class
        "svc": {"class": SVC, "kwargs": {"probability": True}},
        "lr": {"class": LogisticRegression, "kwargs": {}},
        "dt": {"class": DecisionTreeClassifier, "kwargs": {}},
        "voting": [{"class": SVC, "kwargs": {"probability": True}},
                   {"class": LogisticRegression, "kwargs": {}},
                   {"class": DecisionTreeClassifier, "kwargs": {}}],
        "ngram": {"class": OVOSNgramTagger, "kwargs": {}},
    }

    @classmethod
    def _get_clf(cls, clfid: str):
        if clfid not in cls.id2clf:
            raise ValueError(f"invalid classifier: {clfid}")
        data = cls.id2clf[clfid]
        if isinstance(data, list):
            return [d["class"](**d["kwargs"]) for d in data]
        else:
            return data["class"](**data["kwargs"])

    def __init__(self, bus, config=None):
        super().__init__(bus, config)
        clf = self.config.get("classifier", "voting")
        tagger = self.config.get("tagger", "svc")
        self.engines = {lang: JurebesIntentContainer(
            clf=self._get_clf(clf), tagger=self._get_clf(tagger),
            pipeline=self.config.get("features_pipeline", "tfidf_lemma"),
            tagger_pipeline=self.config.get("tagger_pipeline", "naive"),
            fuzzy=self.config.get("fuzzy", False))
            for lang in self.valid_languages}

    # plugin api
    @classproperty
    def matcher_id(self):
        return "jurebes"

    def match(self, utterances, lang, message):
        return self.calc_intent(utterances, lang=lang)

    def train(self):
        self.train()

    # implementation
    def _get_engine(self, lang=None):
        lang = lang or self.lang
        clf = self.config.get("classifier", "voting")
        tagger = self.config.get("tagger", "svc")
        if lang not in self.engines:
            self.engines[lang] = JurebesIntentContainer(
                clf=self._get_clf(clf), tagger=self._get_clf(tagger),
                pipeline=self.config.get("features_pipeline", "tfidf_lemma"),
                tagger_pipeline=self.config.get("tagger_pipeline", "naive"),
                fuzzy=self.config.get("fuzzy", False))
        return self.engines[lang]

    def detach_intent(self, skill_id, intent_name):
        LOG.debug("Detaching jurebes intent: " + intent_name)
        with self.lock:
            munged = _munge(intent_name, skill_id)
            for lang in self.engines:
                self.engines[lang].remove_intent(munged)
        super().detach_intent(skill_id, intent_name)

    def detach_entity(self, skill_id, entity_name):
        LOG.debug("Detaching jurebes entity: " + entity_name)
        with self.lock:
            munged = _munge(entity_name, skill_id)
            for lang in self.engines:
                self.engines[lang].remove_entity(munged)
        super().detach_entity(skill_id, entity_name)

    def detach_skill(self, skill_id):
        LOG.debug("Detaching jurebes skill: " + skill_id)
        with self.lock:
            for lang in self.engines:
                ents = []
                intents = []
                for entity in self.engines[lang].entity_samples.keys():
                    munged = _munge(entity, skill_id)
                    ents.append(munged)
                for intent in self.engines[lang].intent_samples.keys():
                    munged = _munge(intent, skill_id)
                    intents.append(munged)
                for munged in ents:
                    self.engines[lang].remove_entity(munged)
                for munged in intents:
                    self.engines[lang].remove_intent(munged)
        super().detach_skill(skill_id)

    def register_entity(self, skill_id, entity_name, samples=None, lang=None):
        lang = lang or self.lang
        super().register_entity(skill_id, entity_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [entity_name]
        with self.lock:
            container.add_entity(entity_name, samples)

    def register_intent(self, skill_id, intent_name, samples=None, lang=None):
        lang = lang or self.lang
        super().register_intent(skill_id, intent_name, samples, lang)
        container = self._get_engine(lang)
        samples = samples or [intent_name]
        intent_name = _munge(intent_name, skill_id)
        with self.lock:
            container.add_intent(intent_name, samples)

    def calc_intent(self, utterance, min_conf=0.0, lang=None):
        lang = lang or self.lang
        container = self._get_engine(lang)
        min_conf = min_conf or self.config.get("min_conf", 0.35)
        utterance = utterance.strip().lower()
        with self.lock:
            intent = container.calc_intent(utterance).__dict__
        if intent.confidence < min_conf:
            return None

        intent_type, skill_id = _unmunge(intent.intent_name)
        return IntentMatch(intent_service=self.matcher_id,
                           intent_type=intent_type,
                           intent_data=intent.entities,
                           confidence=intent.confidence,
                           utterance=utterance,
                           skill_id=skill_id)
