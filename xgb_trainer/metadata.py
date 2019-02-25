# Data structure of the training file
TRAIN_FILE='data_melanoma_image_train_150x150_train_2.h5'
MODEL_FILE_NAME='melanoma_analysis'

# Specify the learning task, booster type, and verbosity
SILENT = True
OBJECTIVE = 'binary:logistic' # 
BOOSTER = 'gbtree' # Can be gbtree | gblinear | dart

# Choose splitting settings
N_SPLITS = 10
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = None
SHUFFLE = False

# Labels describing features
COLUMNS = list(range(67501))
TARGET = 0
FEATURES = COLUMNS[1:]