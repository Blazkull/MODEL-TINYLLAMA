import os
import zipfile
from tqdm import tqdm

def zip_folder_with_progress(folder_path, output_zip):
    # Obtener lista de todos los archivos y su tamaño total
    files_to_zip = []
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_to_zip.append(file_path)
            total_size += os.path.getsize(file_path)

    # Crear el zip con barra de progreso granular y soporte para archivos grandes (allowZip64=True)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Zipeando modelo") as pbar:
            for file_path in files_to_zip:
                arcname = os.path.relpath(file_path, folder_path)
                
                # Para archivos muy grandes, leemos por trozos (chunks)
                with open(file_path, 'rb') as f_in:
                    # force_zip64=True inside open if needed, but allowZip64=True in the constructor covers it
                    with zf.open(arcname, 'w', force_zip64=True) as f_out:
                        while True:
                            chunk = f_in.read(1024 * 1024) # 1MB chunk
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))

if __name__ == "__main__":
    carpeta = "model_nurse_final"
    zip_name = "model_nurse_final.zip"

    # Borrar zip viejo si existe
    if os.path.exists(zip_name):
        try:
            os.remove(zip_name)
        except PermissionError:
            print(f"Error: No se pudo borrar {zip_name}. ¿Está abierto en otro programa?")
            exit(1)

    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe.")
        exit(1)

    print(f"Iniciando compresión granular con soporte Zip64 de '{carpeta}'...")
    zip_folder_with_progress(carpeta, zip_name)
    print("\n ZIP creado correctamente")
