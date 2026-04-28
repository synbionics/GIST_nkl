import os
import scanpy as sc
import numpy as np
import pandas as pd
from GIST.utils.clustering import *
from GIST.GIST import GIST
import torch
import time
import tracemalloc


# =====================================================================
# IMPOSTAZIONI GLOBALI
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 35
# Percorso base dove si trovano i dati
BASE_DATA_PATH = "/home/nicolae/TESI/GIST_nkl/inputs/spatial_data/Data"

def read_adata(path, is_h5ad=False):
    if is_h5ad:
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()
        adata.obsm["spatial"]=adata.obsm["spatial"].astype(float)
    else: 
        # Carica Visium con il nome file corretto (quello che abbiamo rinominato)
        adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obsm["spatial"]=adata.obsm["spatial"].astype(float)
    return adata


def fromlayerstonumber (layers):
  res=[]
  for sub in layers:
    if sub == 'Layer1':
      res.append(str(sub).replace('Layer1', '1'))
    elif sub == 'Layer2':
      res.append(str(sub).replace('Layer2', '2'))
    elif sub == 'Layer3':
      res.append(str(sub).replace('Layer3', '3'))
    elif sub == 'Layer4':
      res.append(str(sub).replace('Layer4', '4'))
    elif sub == 'Layer5':
      res.append(str(sub).replace('Layer5', '5'))
    elif sub == 'Layer6':
      res.append(str(sub).replace('Layer6', '6'))
    elif sub == 'WM':
      res.append(str(sub).replace('WM', '7'))
    elif str(sub)=='nan' :
      res.append( "-1") ##nan
  return res

def fromlayerstonumberMBA (df_meta_layer):
    label=1
    for i in np.unique(df_meta_layer):
        df_meta_layer[np.where(df_meta_layer == i )[0]]=label
        label+=1
    return df_meta_layer

def fromlayerstonumberMHC(adata):
    adata.obs["ground_truth"] = (
    adata.obs["cluster"].astype("category").cat.codes + 1
).astype(str)
   
def fromlayerstonumberMVC(adata):
   adata.obs["ground_truth"] = (
    adata.obs["label"].astype("category").cat.codes + 1
).astype(str)

def get_adata(path='',data_name='',  is_h5ad=False):
    adata = read_adata(path, is_h5ad)
    print("data name:", data_name)

    if "Human_Breast_Cancer" in data_name :
        df_meta = pd.read_csv(f"{path}/metadata.tsv", sep='\t')
        df_meta_layer = df_meta['fine_annot']
        adata.obs['ground_truth'] =df_meta_layer.values 
        print(f"Data {data_name} contains annotation")
    elif "Mouse_Brain_Anterior" in data_name:
        df_meta = pd.read_csv(f"{path}/metadata.tsv", sep='\t')
        df_meta_layer = df_meta['ground_truth']       
        adata.obs['ground_truth'] = np.array(fromlayerstonumberMBA (df_meta_layer)).astype(str) 
        print(f"Data {data_name} contains annotation")
    elif "DLPFC" in data_name: 
        annotation_path= f"{path}/metadata.tsv"
        df_meta = pd.read_csv(annotation_path, sep='\t')
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = fromlayerstonumber (df_meta_layer.values)  
        print(f"Data {data_name} contains annotation")
    elif "Mouse_Hippocampus" in data_name: 
        fromlayerstonumberMHC (adata)  
        print(f"Data {data_name} contains annotation")
    elif "Mouse_Visual_Cortex" in data_name: 
        fromlayerstonumberMVC (adata)  
        print(f"Data {data_name} contains annotation")
    elif "Mouse_Brain_Ant" in data_name:
        if os.path.exists(f"{path}/metadata.tsv"):
            df_meta = pd.read_csv(f"{path}/metadata.tsv", sep='\t')
            df_meta_layer = df_meta['ground_truth']       
            adata.obs['ground_truth'] = np.array(fromlayerstonumberMBA(df_meta_layer)).astype(str) 
            print(f"Data {data_name} contains annotation")
    elif "Human_Lymph_Node" in data_name or "Mouse_Kidney" in data_name:
        # Solitamente non hanno ground truth standard, saltiamo o cerchiamo se presente
        if 'ground_truth' in adata.obs:
            print(f"Data {data_name} contains annotation")
        else:
            print(f"Data {data_name} does not have annotation")
    elif 'ground_truth' in adata.obs and len(adata.obs['ground_truth']):
         print(f"Data {data_name} contains annotation")
    else: 
        print(f"Data {data_name} does not have annotation")

    return adata

def get_cluster_size(data_name):
    n_cluster=1
    plot_size=0
    if data_name.split("_")[1] in [ '151669', '151670', '151671', '151672']:
        n_cluster= 5
    elif data_name.split("_")[1] in  [ '151507','151508','151509','151510', '151673', '151674', '151675', '151676']:
         n_cluster= 7
    elif "Human_Breast_Cancer" in data_name :
        n_cluster=20
    elif "Mouse_Brain_Anterior" in data_name: 
        n_cluster=52
    elif "Human_Ovarian_Cancer" in data_name: 
        n_cluster=8
    elif "Mouse_Hippocampus" in data_name: 
         n_cluster=14
         plot_size=35
    elif "Olfactory_Bulb" in data_name: 
         n_cluster=7
         plot_size=35
    elif "Mouse_Visual_Cortex" in data_name: 
         n_cluster=7
         plot_size=250
    elif "Human_Lymph_Node" in data_name: 
         n_cluster=8
    elif "Mouse_Kidney" in data_name: 
         n_cluster=7
    elif "Mouse_Brain" in data_name: 
         n_cluster=11
         plot_size=200
    elif "Axolotl_Brain" in data_name: 
         n_cluster=16
         plot_size=35
    return n_cluster, plot_size


# =====================================================================
# CREAZIONE AUTOMATICA DELLA LISTA DATASET (Basata sulle tue cartelle)
# =====================================================================
datasets_to_run = [
    {
        'data_name': 'Mouse_Kidney', 
        'data_type': 'Visium', 
        'refinement': True, 
        'path': f'{BASE_DATA_PATH}/Mouse_Kidney', 
        'is_h5ad': False
    },
    {
        'data_name': 'Mouse_Brain_Ant', 
        'data_type': 'Visium', 
        'refinement': True, 
        'path': f'{BASE_DATA_PATH}/Mouse_Brain_Ant', 
        'is_h5ad': False
    },
    {
        'data_name': 'Human_Lymph_Node', 
        'data_type': 'Visium', 
        'refinement': True, 
        'path': f'{BASE_DATA_PATH}/Human_Lymph_Node', 
        'is_h5ad': False
    }
]

"""
# 1. Generiamo in automatico i percorsi per tutti i 12 campioni DLPFC
dlpfc_samples = ['151507', '151508', '151509', '151510', 
                 '151669', '151670', '151671', '151672', 
                 '151673', '151674', '151675', '151676']

for sample in dlpfc_samples:
    datasets_to_run.append({
        'data_name': f'DLPFC_{sample}', 
        'data_type': 'Visium', 
        'refinement': True, 
        'path': f'inputs/spatial_data/Data/1.DLPFC/{sample}', 
        'is_h5ad': False
    })

# 2. Aggiungiamo Human Breast Cancer
datasets_to_run.append({
    'data_name': 'Human_Breast_Cancer', 
    'data_type': 'Visium', 
    'refinement': True, 
    'path': 'inputs/spatial_data/Data/3.Human_Breast_Cancer', 
    'is_h5ad': False
})

# 3. Aggiungiamo Human Ovarian Cancer
datasets_to_run.append({
    'data_name': 'Human_Ovarian_Cancer', 
    'data_type': 'Visium', 
    'refinement': True, 
    'path': 'inputs/spatial_data/Data/Human_Ovarian_Cancer', 
    'is_h5ad': False
})
"""

# =============================
# MOTORE DI ADDESTRAMENTO 
# =============================
os.makedirs(f"{BASE_DATA_PATH}/Preprocessed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

for ds in datasets_to_run:
    data_name = ds['data_name']
    data_type = ds['data_type']
    refinement = ds['refinement']
    path = ds['path']
    is_h5ad = ds['is_h5ad']
    
    print(f"\n{'='*70}")
    print(f"INIZIO ELABORAZIONE DATASET: {data_name} (MODELLO ORIGINALE)")
    print(f"{'='*70}\n")
    
    # Caricamento Dati
    adata = get_adata(path, data_name, is_h5ad=is_h5ad)
    adata_raw = adata.copy()
    
    # Addestramento GIST
    start_time = time.time()
    tracemalloc.start()

    GISTModel = GIST(adata=adata, device=device, random_seed=seed, data_type=data_type)
    adata = GISTModel.train()

    current, peak = tracemalloc.get_traced_memory()
    end_time = time.time()
    tracemalloc.stop()

    print(f"Tempo di esecuzione ({data_name}): {end_time - start_time:.4f} secondi")
    print(f"Picco memoria ({data_name}): {peak / 10**6:.4f} MB")   

    # Salvataggio Pre-Clustering
    #os.makedirs("inputs/spatial_data/Data/Preprocessed", exist_ok=True)
    #adata.write_h5ad(f"inputs/spatial_data/Data/Preprocessed/{data_name}_originale.h5ad")

    # Clustering e Plot
    n_cluster, plot_size = get_cluster_size(data_name)
    adata = clustering_method(adata, n_pca=20, num_cluster=n_cluster, refinement=refinement, seed=seed)
    
    # SALVATAGGIO IMMAGINE DINAMICO
    nome_immagine = f"outputs/{data_name}_originale.png"
    plot_cluster(adata, nome_immagine, plot_size=plot_size)
    
    # Valutazione Metriche
    metriche = evaluate_cluster(adata, is_visium=GISTModel.is_visium)

    # Salvataggio metriche in file di testo dedicato
    with open("outputs/GIST-original_metriche.txt", "a") as f:
        f.write(f"Dataset: {data_name} | Tempo: {end_time-start_time:.2f}s | Metriche: {metriche}\n")

    # Pulizia e Salvataggio Post-Clustering
    adata_raw.obs['cluster'] = '-1'
    common = adata.obs_names.intersection(adata_raw.obs_names)
    adata_raw.obs.loc[common, 'cluster'] = adata.obs.loc[common, 'cluster'].values
    adata_raw.uns['GIST_emb'] = adata.obsm['GIST_emb']

    adata.write_h5ad(f"inputs/spatial_data/Data/Preprocessed/{data_name}_final_originale.h5ad")
    
    print(f"Dataset {data_name} completato con successo\n")

print("TUTTI I DATASET SONO STATI ELABORATI")