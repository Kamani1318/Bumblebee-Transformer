from model import model 
from Tokenising_Data import training, validation

results = model.train_model(training)

results = model.eval_model(validation)