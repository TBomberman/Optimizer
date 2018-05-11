import deepchem

featurizer = deepchem.feat.WeaveFeaturizer()
loader = deepchem.data.CSVLoader(tasks=['dummy'], smiles_field="canonical_smiles", featurizer=featurizer)
dataset = loader.featurize('LDS-1191/data/merged_id_smiles.csv', shard_size=8192)
