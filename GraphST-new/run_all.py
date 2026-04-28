import os
import time
import tracemalloc
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
import rpy2.rinterface_lib.callbacks

# Silenziamo R per mantenere il terminale pulito
rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, silhouette_score
# Importiamo la metrica custom presa da GIST (Assicurati che il file silhouette_spatial.py sia in questa cartella)
from silhouette_spatial import silhouette_spatial_score
from GraphST import GraphST
from GraphST.utils import clustering

# 2. CONFIGURAZIONE E PERCORSI LINUX/WSL
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Utilizzo del dispositivo: {device}")

base_dir = "/home/nicolae/TESI/GIST_nkl/inputs/spatial_data/Data"
preprocessed_dir = os.path.join(base_dir, "Preprocessed")
dlpfc_samples = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]

# Lista dinamica dei percorsi
datasets = [os.path.join(base_dir, "1.DLPFC", sample) for sample in dlpfc_samples]
datasets.append(os.path.join(base_dir, "3.Human_Breast_Cancer"))
datasets.append(os.path.join(base_dir, "Human_Ovarian_Cancer"))
datasets.append(os.path.join(base_dir, "Human_Lymph_Node"))
datasets.append(os.path.join(base_dir, "Mouse_Brain_Ant"))
datasets.append(os.path.join(base_dir, "Mouse_Kidney"))

# Creiamo la cartella di output per le immagini/log e quella per gli h5ad se non esistono
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(preprocessed_dir, exist_ok=True)

# File per salvare i risultati testuali
results_file = os.path.join(output_dir, "risultati_metriche_GraphST.txt")
with open(results_file, "w") as f:
    f.write("Dataset\tTime(s)\tMemory(MB)\tARI\tAMI\tHomogeneity\tSilhouette\tSSS_Spatial\tSSS_Penalty\n")

# 3. CICLO DI ADDESTRAMENTO
for data_path in datasets:
    dataset_name = os.path.basename(data_path)
    # Rinominiamo i DLPFC per il salvataggio se il nome è solo un numero
    save_name = f"DLPFC_{dataset_name}" if dataset_name.isdigit() else dataset_name
    
    print(f"\n{'='*70}\nINIZIO ELABORAZIONE DATASET: {save_name} (GraphST)\n{'='*70}")
    
    try:
        # Caricamento dati
        adata = sc.read_visium(data_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        
        metadata_path = os.path.join(data_path, 'metadata.tsv')
        has_gt = False
        
        if os.path.exists(metadata_path):
            df_meta = pd.read_csv(metadata_path, sep='\t')
            # Cerchiamo la colonna giusta tra i nomi comuni
            possibili = ['layer_guess', 'fine_annot', 'ground_truth', 'cluster', 'annotation']
            col_name = next((c for c in possibili if c in df_meta.columns), None)
            
            if col_name:
                adata.obs['ground_truth'] = df_meta[col_name].astype(str).values
                # Calcoliamo quanti cluster reali ci sono
                valid_mask = ~pd.isnull(adata.obs['ground_truth']) & (adata.obs['ground_truth'] != 'nan')
                n_clusters = adata.obs['ground_truth'][valid_mask].nunique()
                has_gt = True
                print(f"Ground truth trovato nella colonna: {col_name} ({n_clusters} cluster)")
            else:
                print(f"File metadati trovato ma nessuna colonna nota per {save_name}.")

        # Se non abbiamo trovato il Ground Truth, impostiamo i parametri di emergenza
        if not has_gt:
            print(f"Procedo senza Ground Truth per {save_name}. Uso n_clusters di default.")
            adata.obs['ground_truth'] = "Unknown"
            # Impostiamo n_clusters basandoci sul nome del dataset (come facevi in GIST)
            n_clusters = 8 if "Ovarian_Cancer" in save_name else 7
        
        # Addestramento GraphST
        model = GraphST.GraphST(adata, device=device, epochs=1100)
        
        # Tracciamento Memoria e Tempo
        start_time = time.time()
        tracemalloc.start()
        
        adata = model.train()
        
        #Pulizia preventiva della memoria R prima del clustering
        robjects.r('gc()') # Forza il garbage collector di R

        # Clustering GraphST (usa mclust di default e salva i risultati in adata.obs['domain'])
        clustering(adata, n_clusters, radius=50, method='mclust', refinement=True)
        
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()
        tracemalloc.stop()
        
        exec_time = end_time - start_time
        peak_memory_mb = peak / 10**6
        print(f"Tempo di esecuzione: {exec_time:.2f} s | Picco memoria: {peak_memory_mb:.2f} MB")
        
        # 4. CALCOLO METRICHE (Equivalente a GIST evaluate_cluster)
        # Controlliamo se abbiamo il Ground Truth (GT)
        has_gt = 'ground_truth' in adata.obs and adata.obs['ground_truth'].unique()[0] != "Unknown"
        
        if has_gt:
            valid_idx = ~pd.isnull(adata.obs['ground_truth']) & (adata.obs['ground_truth'] != 'nan')
            gt_labels = adata.obs['ground_truth'][valid_idx]
            pred_labels = adata.obs['domain'][valid_idx]
            
            ari = adjusted_rand_score(gt_labels, pred_labels)
            ami = adjusted_mutual_info_score(gt_labels, pred_labels)
            homog = homogeneity_score(gt_labels, pred_labels)
        else:
            print(f"Dataset {save_name} senza Ground Truth: ARI/AMI/Homog impostati a 0.0")
            ari, ami, homog = 0.0, 0.0, 0.0

        # Metriche INDIPENDENTI dal Ground Truth (Funzionano sempre)
        pca_embeddings = adata.obsm['emb']
        
        # Silhouette standard
        silhouette = silhouette_score(pca_embeddings, adata.obs['domain'], metric='cosine')
        
        # Silhouette Spatial SSS (usando la tua funzione importata da GIST)
        sss_spatial = silhouette_spatial_score(pca_embeddings, adata.obs['domain'], adata, metric="cosine", is_visium=True)
        
        # Recupero Penalty (settato internamente da silhouette_spatial_score)
        sss_penalty = adata.uns.get('average_penalty', 0.0)
        
        print(f"RISULTATI {save_name}: ARI: {ari:.4f} | Silhouette: {silhouette:.4f} | SSS: {sss_spatial:.4f}")
        
        # Salvataggio nel file TXT (mantenendo la struttura delle colonne)
        with open(results_file, "a") as f:
            f.write(f"{save_name}\t{exec_time:.4f}\t{peak_memory_mb:.4f}\t{ari:.4f}\t{ami:.4f}\t{homog:.4f}\t{silhouette:.4f}\t{sss_spatial:.4f}\t{sss_penalty:.4f}\n")
        
        # 5. SALVATAGGIO IMMAGINE
        nome_immagine = os.path.join(output_dir, f"{save_name}_GraphST_Original.png")
        sc.pl.spatial(adata, color=["ground_truth", "domain"], title=[f"GT - {save_name}", f"GraphST (ARI: {ari:.4f})"], show=False)
        plt.savefig(nome_immagine, bbox_inches='tight', dpi=300)
        plt.close()

        # 6. SALVATAGGIO DATI H5AD
        # Copiamo domain in cluster per la compatibilità con MultimetricST
        adata.obs['cluster'] = adata.obs['domain'].copy()
        adata.write_h5ad(os.path.join(preprocessed_dir, f"{save_name}_final_GraphST_Original.h5ad"))
        
        print(f"Salvataggi completati per {save_name}.")
        
    except Exception as e:
        print(f"ERRORE GRAVE sul dataset {save_name}: {e}")

print(f"\nELABORAZIONE COMPLETATA. Risultati tabulari salvati in: {results_file}")

