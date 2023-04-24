from dataclasses import dataclass

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, SklearnOVOSVotingClassifier
from padacioso import IntentContainer as PadaciosoIntentContainer
from sklearn.svm import SVC


@dataclass()
class IntentMatch:
    intent_name: str
    confidence: float
    entities: dict


class JurebesIntentContainer:
    def __init__(self, pipeline, clf=None):
        clf = clf or [SVC(probability=True), LogisticRegression()]
        self.padacioso = PadaciosoIntentContainer()
        if isinstance(clf, list):
            self.jurebes = SklearnOVOSVotingClassifier(clf, pipeline)
        else:
            self.jurebes = SklearnOVOSClassifier(pipeline, clf)
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

    def calc_intents(self, query):
        query = query.lower()
        for exact_intent in self.padacioso.calc_intents(query):
            if exact_intent["name"]:
                yield IntentMatch(confidence=1.0,
                                  intent_name=exact_intent["name"],
                                  entities=exact_intent["entities"])

        prob = self.jurebes.predict_proba([query])[0]
        intent = self.jurebes.predict([query])[0]
        yield IntentMatch(confidence=prob,
                          intent_name=intent,
                          entities={})

    def calc_intent(self, query):
        intents = list(self.calc_intents(query))
        if len(intents):
            return max(intents, key=lambda k: k.confidence)
        return None

    def train(self):
        X, y = self.get_dataset()
        self.jurebes.train(X, y)

    def get_dataset(self):
        X = []
        y = []

        def expand(sample, intent):
            for entity in self.entity_lines:
                tok = "{" + entity + "}"
                if tok not in sample:
                    continue
                for s in self.entity_lines[entity]:
                    X.append(s.replace(tok, s))
                    y.append(intent)

        for intent, samples in self.intent_lines.items():
            for s in self.intent_lines[intent]:
                if "{" in s:
                    expand(s, intent)
                else:
                    X.append(s.lower())
                    y.append(intent)
        return X, y

    def _is_exact(self, intent, sample):
        exact = self.padacioso.calc_intent(sample)
        if exact["name"] is not None:
            if exact["name"] == intent:
                return True
        return False

    def accuracy(self, test_set):
        X = []
        y = []
        for intent, samples in test_set.items():
            for s in samples:
                X.append(s.lower())
                y.append(intent)
        return self.jurebes.score(X, y)


if __name__ == "__main__":
    hello = ["hello human", "hello there", "hey", "hello", "hi"]
    name = ["my name is {name}", "call me {name}", "I am {name}",
            "the name is {name}", "{name} is my name", "{name} is my name"]
    joke = ["tell me a joke", "say a joke", "tell joke"]

    # single clf
    clf = SVC(probability=True)
    # multiple classifiers will use soft voting to select prediction
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    #clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

    engine = JurebesIntentContainer("tfidf_lemma", clf)

    engine.add_entity("name", ["jarbas", "bob"])
    engine.add_intent("hello", hello)
    engine.add_intent("name", name)
    engine.add_intent("joke", joke)

    engine.train()

    test_set = {"name": ["I am groot", "my name is jarbas", "jarbas is the name"],
                "hello": ["hello beautiful", "hello bob", "hello world"],
                "joke": ["say a joke", "make me laugh", "do you know any joke"]}

    print(engine.accuracy(test_set))
    # 0.8888888888888888

    for intent, sents in test_set.items():
        for sent in sents:
            print(sent, engine.calc_intent(sent))
    # I am groot IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'groot'})
    # my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
    # jarbas is the name IntentMatch(intent_name='name', confidence=0.6349775828944594, entities={})
    # hello beautiful IntentMatch(intent_name='hello', confidence=0.7334856864919204, entities={})
    # hello bob IntentMatch(intent_name='hello', confidence=0.7334856864919204, entities={})
    # hello world IntentMatch(intent_name='hello', confidence=0.7334856864919204, entities={})
    # say a joke IntentMatch(intent_name='joke', confidence=1.0, entities={})
    # make me laugh IntentMatch(intent_name='name', confidence=0.42537858099590997, entities={})
    # do you know any joke IntentMatch(intent_name='joke', confidence=0.4801172946911337, entities={})
