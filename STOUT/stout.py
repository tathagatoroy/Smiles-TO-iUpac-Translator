# Initializing and importing necessary libararies
"""  code which contains the inference functions """
import os
import pickle
import re
import pystow
import tensorflow as tf
from rdkit import Chem

from .repack import helper

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Print tensorflow version
print("Tensorflow version: " + tf.__version__)

# Always select a GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Scale memory growth as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set path
default_path = pystow.join("STOUT-V2", "models")

# model download location
MODEL_URL = "https://storage.googleapis.com/decimer_weights/models.zip"
MODEL_PATH = str(default_path) + "/translator_forward/"

# download models to a default location
if not os.path.exists(MODEL_PATH):
    helper.download_trained_weights(MODEL_URL, default_path)

# Load the packed model forward
reloaded_forward = tf.saved_model.load(default_path.as_posix() + "/translator_forward")

# Load the packed model forward
reloaded_reverse = tf.saved_model.load(default_path.as_posix() + "/translator_reverse")


def translate_forward(smiles: str) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens gets detokenized and the final result is returned in a string format.

    Args:
        smiles (str): user input SMILES in string format.

    Returns:
        result (str): The predicted IUPAC names in string format.
    """

    # Load important pickle files which consists the tokenizers and the maxlength setting
    with open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb") as file :
        inp_lang = pickle.load(file)
    with open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb") as file :
        targ_lang = pickle.load(file)
    with open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb") as file :
        inp_max_length = pickle.load(file)
    if len(smiles) == 0:
        return ''
    smiles = smiles.replace('\\/', '/')
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        splitted_list = list(smiles)
        tokenized_smiles = re.sub(r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list)))
        decoded = helper.tokenize_input(tokenized_smiles, inp_lang, inp_max_length)
        result_predited = reloaded_forward(decoded)
        result = helper.detokenize_output(result_predited, targ_lang)
        return result

    return "Could not generate IUPAC name from invalid SMILES."


def translate_reverse(iupacname: str) -> str:
    """Takes user input splits them into words and generates tokens.
    Tokens are then passed to the model and the model predicted tokens are retrieved.
    The predicted tokens gets detokenized and the final result is returned in a string format.

    Args:
        iupacname (str): user input IUPAC names in string format.

    Returns:
        result (str): The predicted SMILES in string format.
    """

    # Load important pickle files which consists the tokenizers and the maxlength setting
    with open(default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb") as file:
        targ_lang = pickle.load(file)
    with open(default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb") as file:
        inp_lang = pickle.load(file)
    with open(default_path.as_posix() + "/assets/max_length_targ.pkl", "rb") as file:
        inp_max_length = pickle.load(file)

    splitted_list = list(iupacname)
    tokenized_iupac_name = " ".join(map(str, splitted_list))
    decoded = helper.tokenize_input(tokenized_iupac_name, inp_lang, inp_max_length)

    result_predited = reloaded_reverse(decoded)
    result = helper.detokenize_output(result_predited, targ_lang)

    return result
