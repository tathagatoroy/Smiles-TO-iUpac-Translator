""" code illustrate prediction capacity of the model for demo purposes """
import time
from stout import translate_forward, translate_reverse

# STOUT - IUPAC name to SMILES example
# file is available in the Github repository
with open("IUPAC_names_test.txt", "r",encoding="utf-8") as file_iupac:
    with  open("SMILES_predictions", "w", encoding="utf-8") as file_out:
        start = time.time()
        for i, line in enumerate(file_iupac):
            iupac_name = line.strip("\n")
            SMILES = translate_reverse(iupac_name)
            file_out.write(SMILES + "\n")
        file_out.flush()
#file_out.close()

# STOUT - SMILES to IUPAC names example
# file is available in the Github repository
#file_smiles = open("SMILES_test.txt", "r", encoding="utf-8")
#file_out = open("IUPAC_predictions", "w", encoding="utf-8")
with open("SMILES_test.txt", "r", encoding="utf-8") as file_smiles:
    with open("IUPAC_predictions", "w", encoding="utf-8") as file_out:
        for i, line in enumerate(file_smiles):
            SMILES = line.strip("\n")
            iupac_name = translate_forward(SMILES)
            file_out.write(iupac_name + "\n")
        file_out.flush()
        #file_out.close()
time_taken = (time.time() - start) / 100
print(f"Time taken for per prediction is {time_taken} sec\n")
