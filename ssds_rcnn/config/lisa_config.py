import os

BASE_PATH="lisa"
ANNOT_PATH = os.path.sep.join([BASE_PATH,"allAnnotations.csv"])

TRAIN_RECORD = os.path.sep.join([BASE_PATH,"record/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH,"record/testing.record"])
#CLASSES_FILE = os.path.sep.join([BASE_PATH,"record/classes.pbtxt"])
CLASSES_FILE ="lisa/record/classes.pbtxt" 

# initialize the test split size
TEST_SIZE = 0.25
# initialize the class labels dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}