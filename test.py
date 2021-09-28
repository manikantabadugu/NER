# STEP 0 - PRE REQUISITES

# python -m spacy download en_core_web_lg

# TBD: Import libraries
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

# TBD: Load preferred model
nlp= spacy.load('en_core_web_lg')

with open("food.txt") as file:
    dataset = file.read()

# TBD: Load the dataset and test it as-is
doc = nlp(dataset)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

words = ['Ketchup','pasta','carrot','pizza','garlic','tomato sauce','basil',
         'carbonara','eggs','cheek fat','pan cakes','parmigiana','eggplant',
        'fettucine','heavy cream','polenta','risotto','espresso','arrosticini','spaghetti',
         'fiorentina steak','pecorino','maccherone',
        'neutella','amero','pistachio','coca-cola','wine','pastiera','watermelon','cappuccino',
        'ice cream','soup','lemon','chocolate',"pineapple"]

train_data = []

with open('food.txt') as file:
    dataset = file.readlines()
    for sentence in dataset:
        print('********')
        print('sentences:', sentence)
        print('********')
        sentence= sentence.lower()
        entities= []
        for word in words:
            word= word.lower()
            if word in sentence:
                start_index= sentence.index(word)
                end_index = len(word)+ start_index
                print('word:', word)
                print('************')
                print('start_index:', start_index)
                print('end_index:', end_index)
                pos = (start_index, end_index, 'FOOD')
                entities.append(pos)
        element = (sentence.rstrip('\n'), {'entities': entities})
        train_data.append(element)
        print('********')
        print('element:', element)
        print('********')

         
        
# STEP 2 - UPDATE MODEL

# TBD: load the needed pipeline
# ner = nlp.get_pipe('ner')
# for _, annotations in train_data:
#     for ent in annotations.get('entities'):
#         ner.add_label(ent[2])
        
# ner.add_label('FOOD')

# TBD: define the annotations

# TBD: train the model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]



# TBD: define the number of iterations, the batch size and the drop according to your experience or using an empirical value
# Train model
with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(30):
        print("Iteration #" + str(iteration))

        # Data shuffle for each iteration
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in spacy.util.minibatch(train_data, size=3):
            for text, annotations in batch:
                # Create an Example object
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.1)
        print("Losses:", losses)

# Save the model
output_dir= path('/NER/')
nlp.to_disk(output_dir)
print('saved')
# TBD:

nlp_updated = spacy.load(output_dir)

# TBD: test with a old sentence
doc= nlp_updated('Alfredo did not invent any pasta!')
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# TBD: test with a new sentence and an old brand
doc= nlp_updated('I dont like spaghetti with mayo')
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])

# TBD: test with a new sentence and a new brand
doc= nlp_updated('burger with cheese is good')
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])