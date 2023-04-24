# Jurebes

J.U.R.E.B.E.S: Joint Universal Rule-based Engine and Bagging Ensemble-based System

This acronym reflects a combined approach of using rule-based techniques along with a bagging ensemble-based approach for intent parsing in the JUREBES engine, written in Python with the use of NLTK and scikit-learn libraries.


## Usage

```python
from jurebes import JurebesIntentContainer


hello = ["hello human", "hello there", "hey", "hello", "hi"]
name = ["my name is {name}", "call me {name}", "I am {name}",
        "the name is {name}", "{name} is my name", "{name} is my name"]
joke = ["tell me a joke", "say a joke", "tell joke"]

# single clf
clf = SVC(probability=True)
# multiple classifiers will use soft voting to select prediction
# clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

#tagger = OVOSNgramTagger(default_tag="O") # classic nltk
#tagger = SVC(probability=True)  # any scikit-learn clf
tagger = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

# pre defined pipelines from ovos-classifiers
clf_pipeline = "tfidf_lemma"
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
# jarbas is the name IntentMatch(intent_name='name', confidence=0.9201351734080562, entities={'name': 'jarbas'})
# call me Ana Ferreira IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'ferreira'})
# hello beautiful IntentMatch(intent_name='hello', confidence=0.8716522106345048, entities={})
# hello bob IntentMatch(intent_name='hello', confidence=0.5400801051648911, entities={'name': 'bob'})
# hello world IntentMatch(intent_name='hello', confidence=0.8716522106345048, entities={})
# say joke IntentMatch(intent_name='joke', confidence=0.9785338275012387, entities={})
# make me laugh IntentMatch(intent_name='name', confidence=0.725778770677012, entities={})
# do you know any joke IntentMatch(intent_name='joke', confidence=0.917960967116358, entities={})


```