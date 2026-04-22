import os
import scanpy as sc
import pandas as pd

# Importazione della metrica dal framework del prof
from Evaluate.evaluate import evaluate_cluster

# Percorsi base
base_input_dir = "Spatial_Clustering_Methods/Data/preprocessed"
output_dir = "Visualize_Scores"

def main():
    print("=== INIZIO VALUTAZIONE MULTI-MODELLO ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutte le sottocartelle dentro 'preprocessed'
    if not os.path.exists(base_input_dir):
        print(f"ERRORE: La cartella {base_input_dir} non esiste.")
        return
        
    cartelle_modelli = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]
    
    if not cartelle_modelli:
        print(f"Nessuna sottocartella trovata in {base_input_dir}")
        return

    all_models_results = [] # Lista per il file master comparativo

    # Ciclo su ogni cartella (es. GIST-new, GraphST-original...)
    for cartella in cartelle_modelli:
        print(f"\n{'='*60}\n-> ELABORAZIONE MODELLO: {cartella}\n{'='*60}")
        model_dir = os.path.join(base_input_dir, cartella)
        
        files = [f for f in os.listdir(model_dir) if f.endswith('.h5ad')]
        if not files:
            print(f"   [!] Nessun file .h5ad trovato in {cartella}. Salto la cartella.")
            continue
            
        model_results = []
        
        for filename in files:
            file_path = os.path.join(model_dir, filename)
            
            # Estraiamo il nome del dataset in modo dinamico
            # Separiamo la stringa usando "_final_" e prendiamo la prima parte
            dataset_name = filename.split('_final_')[0]
            print(f"   Analisi di: {dataset_name}")
            
            try:
                adata = sc.read_h5ad(file_path)
                
                if 'cluster' not in adata.obs:
                    print(f"   [!] Colonna 'cluster' mancante in {dataset_name}. Salto file.")
                    continue
                    
                pred_labels = adata.obs['cluster'].values
                true_labels = adata.obs['ground_truth'].values if 'ground_truth' in adata.obs else None
                
                # Flag Visium per SSS
                visium_flag = True if "DLPFC" in dataset_name or "Human_Breast" in dataset_name else False
                
                # Calcolo metriche
                metrics_dict = evaluate_cluster(
                    adata=adata,
                    pred=pred_labels,
                    ground_truth=true_labels,
                    is_visium=visium_flag,
                    verbose=False
                )
                
                metrics_dict['Dataset'] = dataset_name
                metrics_dict['Method'] = cartella  # Usa il nome esatto della cartella!
                
                model_results.append(metrics_dict)
                all_models_results.append(metrics_dict)
                
            except Exception as e:
                print(f"   [X] Errore critico su {dataset_name}: {e}")

        # Salvataggio del CSV singolo per il modello corrente
        if model_results:
            df_model = pd.DataFrame(model_results)
            # Riordiniamo le colonne
            cols = ['Method', 'Dataset'] + [c for c in df_model.columns if c not in ['Method', 'Dataset']]
            df_model = df_model[cols]
            
            output_csv = os.path.join(output_dir, f"{cartella}_Results.csv")
            df_model.to_csv(output_csv, index=False)
            print(f"\n=> Salvato riepilogo per [{cartella}] in: {output_csv}")

    # Salvataggio del SUPER-FILE con tutti i modelli insieme
    if all_models_results:
        df_all = pd.DataFrame(all_models_results)
        cols = ['Method', 'Dataset'] + [c for c in df_all.columns if c not in ['Method', 'Dataset']]
        df_all = df_all[cols]
        
        master_csv = os.path.join(output_dir, "TUTTI_I_MODELLI_Results.csv")
        df_all.to_csv(master_csv, index=False)
        print(f"\n{'*'*60}\n=> MASTER CSV COMPARATIVO SALVATO CON SUCCESSO:\n   {master_csv}\n{'*'*60}")

if __name__ == "__main__":
    main()