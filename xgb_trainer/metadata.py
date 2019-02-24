# Data structure of the training file
TRAIN_FILE='data_melanoma_image_train_150x150_train_2.h5'
MODEL_FILE_NAME='melanoma_analysis'

TRAIN_TEST_SPLIT = 0.2

COLUMNS = list(range(67501))
TARGET = 0
FEATURES = COLUMNS[1:]