import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, AllChem
import rdkit

def calculate_descriptors_safe(smiles):
    
    try:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.RemoveHs(mol)
        
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp = np.array(fingerprint)
        fp_array = np.array(fp, dtype=int).reshape(1, -1)
        fp_df = pd.DataFrame(fp_array,
                            columns=[f'fp_{i}' for i in range(fp_array.shape[1])]
        )

        desc = {
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'mw': Descriptors.MolWt(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rot': Descriptors.NumRotatableBonds(mol),
            'logp': Crippen.MolLogP(mol),
            'ring_count': rdMolDescriptors.CalcNumRings(mol)
        }
        desc_df = pd.DataFrame([desc])
        
        final_df = pd.concat([desc_df, fp_df], axis=1)

        return final_df

    except Exception:
        return None

if __name__ == "__main__":
    test_smiles = "COc1cc(OC)cc(-c2cc3cnc(N(CCCO)C(C)=O)cc3nc2NC(=O)NC(C)(C)C)c1"
    descriptors = calculate_descriptors_safe(test_smiles)
    print(descriptors)
    print(rdkit.__version__)