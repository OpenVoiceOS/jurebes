# jurebes

J.U.R.E.B.E.S: Joint Universal Rule-based Engine and Bagging Ensemble-based System

This acronym reflects a combined approach of using rule-based techniques along with a bagging ensemble-based approach for intent parsing in the JUREBES engine, written in Python with the use of NLTK and scikit-learn libraries.


## Usage

```python
from jurebes import JurebesIntentContainer


hello = ["hello human", "hello there", "hey", "hello", "hi"]
name = ["my name is {name}", "call me {name}", "I am {name}",
        "the name is {name}", "{name} is my name", "{name} is my name"]
joke = ["tell me a joke", "say a joke", "tell joke"]


engine = JurebesIntentContainer()

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

## Advanced Usage

you can select the classifiers or enable fuzzy matching and influence predictions, jurebes is stateful

```python
from jurebes import JurebesIntentContainer

# single clf
clf = SVC(probability=True)  # any scikit-learn clf
# multiple classifiers will use soft voting to select prediction
# clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()] / default if not in args

tagger = OVOSNgramTagger(default_tag="O") # classic nltk / default if not in args
#tagger = SVC(probability=True)  # any scikit-learn clf
#tagger = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

# pre defined pipelines from ovos-classifiers
clf_pipeline = "tfidf"  # default if not in args
tagger_pipeline = "words"  # default if not in args
engine = JurebesIntentContainer(clf, tagger,
                                clf_pipeline, tagger_pipeline)

(...)  # register intents

# fuzzy matching
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


# force correct predictions
engine.exclude_keywords("name", ["laugh"])
print(engine.calc_intent("make me laugh"))
# IntentMatch(intent_name='joke', confidence=0.5125373111690074, entities={})
engine.exclude_keywords("hello", ["laugh"])
print(engine.calc_intent("make me laugh"))
# IntentMatch(intent_name='joke', confidence=1.0, entities={})


# inject context
engine.set_context("joke", "joke_type", "chuck_norris")  # if a value is passed it will populate entities
print(engine.calc_intent("tell me a chuch norris joke"))
# IntentMatch(intent_name='joke', confidence=0.9707841337857908, entities={'joke_type': 'chuck_norris'})


# require context
engine.require_context("joke", "joke_type")
engine.unset_context("joke", "joke_type")
print(engine.calc_intent("tell me a chuch norris joke"))
# IntentMatch(intent_name='hello', confidence=0.060199275248566525, entities={})
engine.unrequire_context("joke", "joke_type")
print(engine.calc_intent("tell me a chuch norris joke"))
# IntentMatch(intent_name='joke', confidence=0.9462089582801377, entities={})


# exclude intent matches based on context
engine.exclude_context("hello", "said_hello")
print(engine.calc_intent("hello"))
# IntentMatch(intent_name='hello', confidence=1, entities={})
engine.set_context("hello", "said_hello")  # now wont predict hello intent
print(engine.calc_intent("hello"))
# IntentMatch(intent_name='joke', confidence=0.06986199472674888, entities={})
```