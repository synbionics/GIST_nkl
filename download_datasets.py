import os
import requests
import tarfile
import zipfile
from tqdm import tqdm

# =====================================================================
# CONFIGURAZIONE PERCORSI E URL
# =====================================================================

# cartella base in cui salvare i dati
BASE_DIR = "."

# Se il link è un file .zip o .tar.gz, lo script lo estrarrà in automatico.
DATASET_URLS = {
    # DLPFC 
    "DPLFC": "https://zenodo.org/records/15277298/files/DLPFC.zip?download=1",
    
    # ALTRI DATASET
    "Human_Lymph_Node": "https://zenodo.org/records/15277298/files/Human_lymph_Node.zip?download=1",
    "Human_Breast_Cancer": "https://zenodo.org/records/15277298/files/Human_Breast_Cancer.zip?download=1",
    "Human_Ovarian_Cancer": "https://zenodo.org/records/15277298/files/Human_Ovarian_Cancer.zip?download=1"
}

# =====================================================================
# FUNZIONI DI SUPPORTO
# =====================================================================
def download_file(url, dest_path):
    """Scarica un file da un URL mostrando una progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status() # Lancia un errore se il link non è valido
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as t:
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)

def extract_file(file_path, extract_to):
    """Estrae archivi .zip o .tar.gz nella cartella di destinazione."""
    if file_path.endswith('.zip'):
        print(f"[*] Estrazione di {os.path.basename(file_path)}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        print(f"[*] Estrazione di {os.path.basename(file_path)}...")
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)

# =====================================================================
# ESECUZIONE PRINCIPALE
# =====================================================================
def main():
    print("="*50)
    print("?? INIZIO DOWNLOAD AUTOMATICO DATASET SPATIAL")
    print("="*50)
    
    # Crea la cartella base se non esiste
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Cartella base impostata: {BASE_DIR}\n")

    for dataset_name, url in DATASET_URLS.items():
        print(f"\n--- Processando Dataset: {dataset_name} ---")
        
        # Crea la sottocartella specifica per il dataset (es. Data/151507)
        dataset_dir = os.path.join(BASE_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Determina il nome del file da scaricare
        file_name = url.split('/')[-1]
        file_path = os.path.join(dataset_dir, file_name)
        
        # Controllo se esiste già per evitare di riscaricarlo
        if os.path.exists(file_path):
            print(f"[OK] Il file {file_name} esiste già. Passo al prossimo.")
            continue
            
        try:
            # 1. Scarica il file
            download_file(url, file_path)
            
            # 2. Estrai il file (se è un archivio compress)
            if file_path.endswith(('.zip', '.tar.gz', '.tgz')):
                extract_file(file_path, dataset_dir)
                
        except Exception as e:
            print(f"[ERRORE] Impossibile scaricare o estrarre {dataset_name}: {e}")

    print("\n" + "="*50)
    print("? PROCESSO COMPLETATO!")
    print("="*50)

if __name__ == "__main__":
    main()