import os
import time

from upstock.config import supabase

# supabase storage
def download_model_file():
    bucket_name = 'sentiment_file'
    
    file_paths = [
        'upstock_sentiment_model.keras',
        'upstock_sentiment_model.h5',
        'upstock_sentiment_tokenizer.pickle'
    ]

    os.makedirs('SaveModel', exist_ok=True) # exist no error
    bucket = supabase.storage.from_(bucket_name)
    
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

                print(f"{file_path} download complete {local_path}")
                break

            except Exception as e:
                print(f"{file_path} download fail : {e}")
                if attempt < retries -1:
                    print('retry')
                    time.sleep(3)
                else:
                    print('error')
