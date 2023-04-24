# Jurebes

J.U.R.E.B.E.S. - Just Understand, Recognize, and Extract Byword Entities System


## Usage

```python
from jurebes import JurebesIntentContaine


hello = ["hello human", "hello there", "hey", "hello", "hi"]
name = ["my name is {name}", "call me {name}", "I am {name}",
        "the name is {name}", "{name} is my name", "{name} is my name"]
joke = ["tell me a joke", "say a joke", "tell joke"]

# single clf
clf = SVC(probability=True)
# multiple classifiers will use soft voting to select prediction
# clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

engine = JurebesIntentContainer("tfidf_lemma", clf)

engine.add_entity("name", ["jarbas", "bob", "João Casimiro Ferreira"])
engine.add_intent("hello", hello)
engine.add_intent("name", name)
engine.add_intent("joke", joke)

engine.train()

test_set = {"name": ["I am groot", "my name is jarbas", "jarbas is the name"],
            "hello": ["hello beautiful", "hello bob", "hello world"],
            "joke": ["say a joke", "make me laugh", "do you know any joke"]}

for intent, sents in test_set.items():
    for sent in sents:
        print(sent, engine.calc_intent(sent))

# I am groot IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'groot'})
# my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
# jarbas is the name IntentMatch(intent_name='name', confidence=0.8853768669468998, entities={'name': 'jarbas'})
# hello beautiful IntentMatch(intent_name='hello', confidence=0.8094038390048462, entities={})
# hello bob IntentMatch(intent_name='hello', confidence=0.4518911986498847, entities={'name': 'bob'})
# hello world IntentMatch(intent_name='hello', confidence=0.8094038390048462, entities={})
# say a joke IntentMatch(intent_name='joke', confidence=1.0, entities={})
# make me laugh IntentMatch(intent_name='name', confidence=0.695092259467801, entities={})
# do you know any joke IntentMatch(intent_name='joke', confidence=0.9817285368200765, entities={})

```