"""
from Supabase storage 
"""

import gzip
import shutil
import os
import time
import logging

from upstock.config import supabase, paths, SupabaseConfig

logger = logging.getLogger(__name__)

# supabase storage
def download_model_file():
    
    file_paths = [
        paths.model,
        paths.model_h5,
        paths.tokenizer
    ]

    os.makedirs('SaveModel', exist_ok=True) # exist no error
    bucket = supabase.storage.from_(SupabaseConfig.bucket_name)
    
    for file_path in file_paths:
        
        retries = 3 # 2 time retry
        
        for attempt in range(retries):
            try:
                res = bucket.download(file_path)

                # download() result bytes > res 사용, result response > res.read()실행
                content = res.read() if hasattr(res, "read") else res
                local_path = os.path.join("SaveModel", os.path.basename(file_path))
                
                with open(local_path, "wb") as f:
                    f.write(content)

                logger.info(f"{file_path} download complete {local_path}")
                
                # gzip -9 upstock_sentiment_model.keras
                if local_path.endswith(".gz"): # 압축 제거
                    uncompressed_path = local_path[:-3]  # .gz 제거
                    with gzip.open(local_path, "rb") as f_in:
                        with open(uncompressed_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    logger.info(f"{file_path} decompressed to {uncompressed_path}")
                    
                break

            except Exception as e:
                logger.error(f"{file_path} download failed : {e}")
                if attempt < retries -1:
                    logger.info('Retry download')
                    time.sleep(3)
                else:
                    logger.error(f'{file_path} download failed after {retries} attempts')