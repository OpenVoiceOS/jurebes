from dataclasses import dataclass

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, SklearnOVOSVotingClassifier
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger, SklearnOVOSVotingClassifierTagger
from ovos_classifiers.tasks.tagger import OVOSNgramTagger
from padacioso import IntentContainer as PadaciosoIntentContainer
from quebra_frases import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass()
class IntentMatch:
    intent_name: str
    confidence: float
    entities: dict


class JurebesIntentContainer:
    def __init__(self, clf=None, tagger=None, pipeline="tfidf_lemma", tagger_pipeline="naive", fuzzy=False):
        clf = clf or [SVC(probability=True),
                      LogisticRegression(),
                      DecisionTreeClassifier()]
        self.padacioso = PadaciosoIntentContainer(fuzz=fuzzy)

        if isinstance(clf, list):
            self.classifier = SklearnOVOSVotingClassifier(clf, pipeline)
        else:
            self.classifier = SklearnOVOSClassifier(pipeline, clf)

        if tagger is None:
            self.tagger = OVOSNgramTagger(default_tag="O")
        elif isinstance(tagger, list):
            self.tagger = SklearnOVOSVotingClassifierTagger(tagger, tagger_pipeline)
        elif isinstance(tagger, OVOSNgramTagger):
            self.tagger = tagger
        else:
            self.tagger = SklearnOVOSClassifierTagger(tagger, tagger_pipeline)
        self.intent_lines, self.entity_lines = {}, {}

    def add_intent(self, name, lines):
        lines = [l.lower() for l in lines]
        self.padacioso.add_intent(name, lines)
        self.intent_lines[name] = lines

    def remove_intent(self, name):
        self.padacioso.remove_intent(name)
        if name in self.intent_lines:
            del self.intent_lines[name]

    def add_entity(self, name, lines):
        lines = [l.lower() for l in lines]
        self.padacioso.add_entity(name, lines)
        self.entity_lines[name] = lines

    def remove_entity(self, name):
        self.padacioso.remove_entity(name)
        if name in self.entity_lines:
            del self.entity_lines[name]

    def get_entities(self, query):
        entities = {}
        in_entity = False
        if isinstance(self.tagger, OVOSNgramTagger):
            tags = self.tagger.tag(query)
        else:
            toks = query.split()
            tags = list(zip(toks, self.tagger.tag(toks)))

        for word, tag in tags:
            if tag == "O":
                in_entity = False
                continue
            ent_name = tag.split("-")[-1]
            if not in_entity:
                if ent_name in entities:
                    pass  # TODO - duplicate entity what do
                entities[ent_name] = word
                in_entity = True
            elif in_entity:
                entities[ent_name] += " " + word

        return entities

    def calc_intents(self, query):
        query = query.lower()
        for exact_intent in self.padacioso.calc_intents(query):
            if exact_intent["name"]:
                yield IntentMatch(confidence=exact_intent["conf"],
                                  intent_name=exact_intent["name"],
                                  entities=exact_intent["entities"])

        prob = self.classifier.predict_proba([query])[0]
        intent = self.classifier.predict([query])[0]
        yield IntentMatch(confidence=prob,
                          intent_name=intent,
                          entities=self.get_entities(query))

    def calc_intent(self, query):
        intents = list(self.calc_intents(query))
        if len(intents):
            return max(intents, key=lambda k: k.confidence)
        return None

    @staticmethod
    def _transform_iob_to_dataset(tagged_sentences):
        X, y = [], []

        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                X.append(tagged[index][0])
                y.append(tagged[index][1])

        return X, y

    def train(self):
        X, y = self.get_dataset()
        self.classifier.train(X, y)

        X = self.get_iob_dataset()
        if isinstance(self.tagger, OVOSNgramTagger):
            self.tagger.train(X)
        else:
            X, y = self._transform_iob_to_dataset(X)
            self.tagger.train(X, y)

    def get_dataset(self):
        X = []
        y = []

        def expand(sample, intent):
            for entity in self.entity_lines:
                tok = "{" + entity + "}"
                if tok not in sample:
                    continue
                for s in self.entity_lines[entity]:
                    X.append(sample.replace(tok, s))
                    y.append(intent)

        for intent, samples in self.intent_lines.items():
            for s in self.intent_lines[intent]:
                if "{" in s:
                    expand(s, intent)
                else:
                    X.append(s.lower())
                    y.append(intent)
        return X, y

    def get_iob_dataset(self):
        X = []

        def expand(sample):
            toks = sample.split(" ")

            for entity in self.entity_lines:
                tok = "{" + entity + "}"
                if tok not in toks:
                    continue
                for s in self.entity_lines[entity]:
                    idx = toks.index(tok)
                    nt = word_tokenize(s)

                    pt1 = toks[:idx]
                    pt2 = toks[idx + 1:]
                    toks2 = pt1 + nt + pt2
                    if len(nt) == 1:
                        iob2 = ["O"] * len(pt1) + [f"B-{entity}"] + ["O"] * len(pt2)
                    else:
                        iob2 = ["O"] * len(pt1) + [f"B-{entity}"] + [f"I-{entity}"] * (len(nt) - 1) + ["O"] * len(pt2)

                    X.append(list(zip(toks2, iob2)))

        for intent, samples in self.intent_lines.items():
            for s in self.intent_lines[intent]:
                toks = word_tokenize(s)
                if "{" in s:
                    expand(s)
                else:
                    iob = ["O"] * len(toks)
                    X.append(list(zip(toks, iob)))

        return X


if __name__ == "__main__":
    hello = ["hello human", "hello there", "hey", "hello", "hi"]
    name = ["my name is {name}", "call me {name}", "I am {name}",
            "the name is {name}", "{name} is my name", "{name} is my name"]
    joke = ["tell me a joke", "say a joke", "tell joke"]

    # single clf
    clf = SVC(probability=True)
    # multiple classifiers will use soft voting to select prediction
    # clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

    # tagger = OVOSNgramTagger(default_tag="O") # classic nltk
    # tagger = SVC(probability=True)  # any scikit-learn clf
    tagger = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

    # pre defined pipelines from ovos-classifiers
    clf_pipeline = "tfidf"
    tagger_pipeline = "words"
    engine = JurebesIntentContainer(clf, tagger,
                                    clf_pipeline, tagger_pipeline)

    engine.add_entity("name", ["jarbas", "bob", "João Casimiro Ferreira"])
    engine.add_intent("hello", hello)
    engine.add_intent("name", name)
    engine.add_intent("joke", joke)

    engine.train()

    test_set = {"name": ["I am groot", "my name is jarbas",
                         "jarbas is the name", "they call me Ana Ferreira"],
                "hello": ["hello beautiful", "hello bob", "hello world"],
                "joke": ["say joke", "make me laugh", "do you know any joke"]}

    for intent, sents in test_set.items():
        for sent in sents:
            print(sent, engine.calc_intent(sent))

    # I am groot IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'groot'})
    # my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
    # jarbas is the name IntentMatch(intent_name='name', confidence=0.9141868974242002, entities={'name': 'jarbas'})
    # they call me Ana Ferreira IntentMatch(intent_name='name', confidence=0.8599902513687149, entities={'name': 'ferreira'})
    # hello beautiful IntentMatch(intent_name='hello', confidence=0.8475377118056997, entities={})
    # hello bob IntentMatch(intent_name='hello', confidence=0.5025161990154762, entities={'name': 'bob'})
    # hello world IntentMatch(intent_name='hello', confidence=0.8475377118056997, entities={})
    # say joke IntentMatch(intent_name='joke', confidence=0.9785338275012387, entities={})
    # make me laugh IntentMatch(intent_name='name', confidence=0.6135639618555918, entities={})
    # do you know any joke IntentMatch(intent_name='joke', confidence=0.9785338275012387, entities={})
