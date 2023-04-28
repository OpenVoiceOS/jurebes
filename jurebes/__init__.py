from dataclasses import dataclass

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, SklearnOVOSVotingClassifier
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifierTagger, SklearnOVOSVotingClassifierTagger
from ovos_classifiers.tasks.tagger import OVOSNgramTagger
from ovos_utils.log import LOG
from padacioso import IntentContainer as PadaciosoIntentContainer
from padacioso.bracket_expansion import expand_parentheses
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
        self.intent_samples, self.entity_samples = {}, {}
        self.available_contexts = {}
        self.required_contexts = {}
        self.excluded_keywords = {}
        self.excluded_contexts = {}
        self.detached_intents = []
        self.detached_entities = []

    def enable_fuzzy(self):
        self.padacioso.fuzz = True

    def disable_fuzzy(self):
        self.padacioso.fuzz = False

    def add_intent(self, intent_name, samples):
        expanded = []
        for l in samples:
            l = l.replace("{{", "{").replace("}}", "}")
            expanded += expand_parentheses(l.lower())
        samples = list(set(expanded))

        self.padacioso.add_intent(intent_name, samples)
        self.intent_samples[intent_name] = samples

    def detach_intent(self, intent_name):
        if intent_name not in self.detached_intents:
            self.detached_intents.append(intent_name)

    def reatach_intent(self, intent_name):
        if intent_name in self.detached_intents:
            self.detached_intents.remove(intent_name)

    def detach_entity(self, entity_name):
        if entity_name not in self.detached_entities:
            self.detached_entities.append(entity_name)

    def reatach_entity(self, entity_name):
        if entity_name in self.detached_entities:
            self.detached_entities.remove(entity_name)

    def exclude_keywords(self, intent_name, samples):
        if intent_name not in self.excluded_keywords:
            self.excluded_keywords[intent_name] = samples
        else:
            self.excluded_keywords[intent_name] += samples

    def set_context(self, intent_name, context_name, context_val=None):
        if intent_name not in self.available_contexts:
            self.available_contexts[intent_name] = {}
        self.available_contexts[intent_name][context_name] = context_val

    def exclude_context(self, intent_name, context_name):
        if intent_name not in self.excluded_contexts:
            self.excluded_contexts[intent_name] = [context_name]
        else:
            self.excluded_contexts[intent_name].append(context_name)

    def unexclude_context(self, intent_name, context_name):
        if intent_name in self.excluded_contexts:
            self.excluded_contexts[intent_name] = [c for c in self.excluded_contexts[intent_name]
                                                   if context_name != c]

    def unset_context(self, intent_name, context_name):
        if intent_name in self.available_contexts:
            if context_name in self.available_contexts[intent_name]:
                self.available_contexts[intent_name].pop(context_name)

    def require_context(self, intent_name, context_name):
        if intent_name not in self.required_contexts:
            self.required_contexts[intent_name] = [context_name]
        else:
            self.required_contexts[intent_name].append(context_name)

    def unrequire_context(self, intent_name, context_name):
        if intent_name in self.required_contexts:
            self.required_contexts[intent_name] = [c for c in self.required_contexts[intent_name]
                                                   if context_name != c]

    def remove_intent(self, intent_name):
        self.detach_intent(intent_name)
        self.padacioso.remove_intent(intent_name)
        if intent_name in self.intent_samples:
            del self.intent_samples[intent_name]

    def add_entity(self, entity_name, samples):
        expanded = []
        for l in samples:
            expanded += expand_parentheses(l.lower())
        samples = list(set(expanded))
        self.padacioso.add_entity(entity_name, samples)
        self.entity_samples[entity_name] = samples

    def remove_entity(self, entity_name):
        self.detach_entity(entity_name)
        self.padacioso.remove_entity(entity_name)
        if entity_name in self.entity_samples:
            del self.entity_samples[entity_name]

    def get_entities(self, query):
        entities = {}
        in_entity = False
        try:
            if isinstance(self.tagger, OVOSNgramTagger):
                tags = self.tagger.tag(query)
            else:
                toks = query.split()
                tags = list(zip(toks, self.tagger.tag(toks)))
        except Exception as e:
            LOG.error(f"failed to tag entities: {e}")
            return {}

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

        excluded_intents = self.detached_intents
        for intent_name, samples in self.excluded_keywords.items():
            if any(s in query for s in samples):
                excluded_intents.append(intent_name)
        for intent_name, contexts in self.required_contexts.items():
            if intent_name not in self.available_contexts:
                excluded_intents.append(intent_name)
            elif any(context not in self.available_contexts[intent_name]
                     for context in contexts):
                excluded_intents.append(intent_name)

        for intent_name, contexts in self.excluded_contexts.items():
            if intent_name not in self.available_contexts:
                continue
            if any(context in self.available_contexts[intent_name]
                   for context in contexts):
                excluded_intents.append(intent_name)

        ents = self.get_entities(query)

        exact_intents = []

        for exact_intent in self.padacioso.calc_intents(query):
            if exact_intent["name"]:
                if exact_intent["name"] in excluded_intents:
                    continue
                # inject regex extracted entities
                if self.padacioso.fuzz or exact_intent["conf"] >= 0.8:
                    ents.update(exact_intent["entities"])
                exact_intents.append(IntentMatch(confidence=exact_intent["conf"],
                                                 intent_name=exact_intent["name"],
                                                 entities=exact_intent["entities"]))

        for intent_name, samples in self.intent_samples:
            if any(s == query for s in samples if "{" not in s):
                # update confidence of previous match
                for idx, i in enumerate(exact_intents):
                    if i.intent_name == intent_name:
                        exact_intents[idx].confidence = 1
                        break
                # add new exact match
                else:
                    exact_intents.append(IntentMatch(confidence=1.0,
                                                     intent_name=intent_name,
                                                     entities={}))

        ents = {k: v for k, v in ents.items() if k not in self.detached_entities}

        classified_intents = []
        leftover_prob = 0  # redistribute prob of excluded intents so predictions still sum up to 1

        classes = self.classifier.clf.classes_

        if self.classifier.clf is not None:  # trained!
            probs = self.classifier.clf.predict_proba([query])[0]

            for intent, prob in zip(classes, probs):
                if intent in excluded_intents:
                    leftover_prob += prob
                    continue
                if intent in self.available_contexts:
                    for context, val in self.available_contexts[intent].items():
                        if val is not None:
                            ents[context] = val

                for intent in exact_intents:
                    if intent.intent_name != intent:
                        continue
                    # padacioso has a fake score, not a probability
                    # usually returns 1.0 for exact matches
                    # there is some variance between 0.75 and 0.95 with non-exact matches
                    conf2 = exact_intents[intent].confidence
                    ents2 = exact_intents[intent].entities

                    # let's increase the base prediction probability
                    # since we got 2 matches for same intent with different engines
                    if conf2 == 1.0:
                        # sample likely in training set
                        bonus = 1.0
                        # regex capture group match!
                        if ents2:
                            bonus = prob * 0.7
                    else:
                        bonus = conf2 * 0.5

                    prob = min(1.0, prob + bonus)

                exact_intents = [i for i in exact_intents if i.intent_name != intent]

                classified_intents.append(IntentMatch(confidence=prob,
                                                      intent_name=intent,
                                                      entities=ents))

        # redistribute leftover probabilities due to excluded intents
        # increase probs so they sum up to 1
        # excludes/context rules would filter intents but not adjust probabilities
        intents = classified_intents + exact_intents
        total = sum(i.confidence for i in classified_intents)
        normbonus = (1 - total) / len(intents)

        for i in intents:
            i.confidence += normbonus
            yield i

    def calc_intent(self, query, threshold=0):
        intents = list(self.calc_intents(query))
        if len(intents):
            best = max(intents, key=lambda k: k.confidence)
            if best.confidence >= threshold:
                return best
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
        if len(set(y)) < 2:
            LOG.warning("not enough data to train! exact matches only")
        else:
            self.classifier.train(X, y)

        X = self.get_iob_dataset()
        if len(X):
            if isinstance(self.tagger, OVOSNgramTagger):
                self.tagger.train(X)
            else:
                X, y = self._transform_iob_to_dataset(X)
                self.tagger.train(X, y)

    def get_dataset(self):
        X = []
        y = []

        def expand(sample, intent):
            for entity in self.entity_samples:
                tok = "{" + entity + "}"
                if tok not in sample:
                    continue
                for s in self.entity_samples[entity]:
                    X.append(sample.replace(tok, s))
                    y.append(intent)

        for intent, samples in self.intent_samples.items():
            for s in self.intent_samples[intent]:
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

            for entity in self.entity_samples:
                tok = "{" + entity + "}"
                if tok not in toks:
                    continue
                for s in self.entity_samples[entity]:
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

        for intent, samples in self.intent_samples.items():
            for s in self.intent_samples[intent]:
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
                                    clf_pipeline, tagger_pipeline,
                                    fuzzy=False)  # fuzzy match may improve entity extraction

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

    # I am groot IntentMatch(intent_name='name', confidence=0.95, entities={'name': 'groot'})
    # my name is jarbas IntentMatch(intent_name='name', confidence=1, entities={'name': 'jarbas'})
    # jarbas is the name IntentMatch(intent_name='name', confidence=0.9103060265385112, entities={'name': 'jarbas'})
    # they call me Ana Ferreira IntentMatch(intent_name='name', confidence=0.8539103439781974, entities={'name': 'ferreira'})
    # hello beautiful IntentMatch(intent_name='hello', confidence=0.8642926826931172, entities={})
    # hello bob IntentMatch(intent_name='hello', confidence=0.5378103564074476, entities={'name': 'bob'})
    # hello world IntentMatch(intent_name='hello', confidence=0.8642926826931172, entities={})
    # say joke IntentMatch(intent_name='joke', confidence=0.9409318377180504, entities={})
    # make me laugh IntentMatch(intent_name='name', confidence=0.6116405452151128, entities={})
    # do you know any joke IntentMatch(intent_name='joke', confidence=0.9409318377180504, entities={})

    engine.enable_fuzzy()
    sent = "they call me Ana Ferreira"
    print(engine.calc_intent(sent))
    # IntentMatch(intent_name='name', confidence=0.8716633619210677, entities={'name': 'ana ferreira'})
    engine.disable_fuzzy()
    print(engine.calc_intent(sent))
    # IntentMatch(intent_name='name', confidence=0.8282293617609358, entities={'name': 'ferreira'})

    # temporarily disable a intent
    engine.detach_intent("name")
    print(engine.calc_intent(sent))
    # IntentMatch(intent_name='hello', confidence=0.06113697262028985, entities={'name': 'ferreira'})
    engine.reatach_intent("name")
    print(engine.calc_intent(sent))
    # IntentMatch(intent_name='name', confidence=0.8548664325189478, entities={'name': 'ferreira'})

    engine.exclude_keywords("name", ["laugh"])
    print(engine.calc_intent("make me laugh"))
    # IntentMatch(intent_name='joke', confidence=0.5125373111690074, entities={})

    engine.exclude_keywords("hello", ["laugh"])
    print(engine.calc_intent("make me laugh"))
    # IntentMatch(intent_name='joke', confidence=1.0, entities={})
