import simpletransformers
import logging
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#model args to be used
model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 41,
    "train_batch_size": 10,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "manual_seed": 4,
}

encoder_type = "roberta"


#The 1st 3 parameters are model type, encoder and decoder exact model name to be used
model = Seq2SeqModel(
    encoder_type,
    "roberta-base",
    "bert-base-cased",
    args=model_args,
    use_cuda=True,
)