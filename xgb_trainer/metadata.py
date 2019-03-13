# Data structure of the training file
TRAIN_FILE='melanoma_image_train_9792x67501.h5'
MODEL_FILE_NAME='isicTrainer'

# Specify the learning task, booster type, and verbosity
SILENT = False
OBJECTIVE = 'binary:logistic' # 
BOOSTER = 'gbtree' # Can be gbtree | gblinear | dart
N_JOBS = -1
EARLY_STOPPING_ROUNDS = 100

# Choose splitting settings
N_SPLITS = 10
TRAIN_TEST_SPLIT = 0.2
SCORING = 'accuracy'
RANDOM_STATE = None
SHUFFLE = False

# Labels describing features
COLUMNS = list(range(67501))
TARGET = 0
FEATURES = COLUMNS[1:]