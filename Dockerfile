# ------------------------------------------------------------
# Base
# ------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# 빌드/런타임 최소 deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python deps
#  - 제공하신 패키지 버전을 그대로 사용
#  - tensorflow-macos는 주석 처리(리눅스 컨테이너에서 미지원)
# ------------------------------------------------------------
RUN <<'REQ' bash -lc 'cat > /tmp/requirements.txt <<EOF
absl-py==2.3.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.13
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
astunparse==1.6.3
attrs==25.3.0
beautifulsoup4==4.13.4
cachetools==5.5.2
certifi==2025.6.15
cffi==1.17.1
charset-normalizer==3.4.2
curl_cffi==0.11.4
dataclasses-json==0.6.7
flatbuffers==25.2.10
frozendict==2.4.6
frozenlist==1.7.0
gast==0.6.0
google-auth==2.40.3
google-auth-oauthlib==1.2.2
google-pasta==0.2.0
grpcio==1.73.1
h11==0.16.0
h5py==3.14.0
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.1
idna==3.10
joblib==1.5.1
jsonpatch==1.33
jsonpointer==3.0.0
kagglehub==0.3.12
keras==2.15.0
langchain==0.3.26
langchain-community==0.3.26
langchain-core==0.3.66
langchain-text-splitters==0.3.8
langsmith==0.4.3
libclang==18.1.1
Markdown==3.8.2
markdown-it-py==3.0.0
MarkupSafe==3.0.2
marshmallow==3.26.1
mdurl==0.1.2
ml-dtypes==0.2.0
multidict==1.7.0
multitasking==0.0.11
mypy_extensions==1.1.0
namex==0.1.0
numpy==1.26.4
oauthlib==3.3.1
opt_einsum==3.4.0
optree==0.16.0
orjson==3.10.18
packaging==24.2
pandas==2.3.0
peewee==3.18.1
platformdirs==4.3.8
propcache==0.3.2
protobuf==4.25.8
pyasn1==0.6.1
pyasn1_modules==0.4.2
pycparser==2.22
pydantic==2.11.7
pydantic-settings==2.10.1
pydantic_core==2.33.2
Pygments==2.19.2
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytz==2025.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.4
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
rich==14.0.0
rsa==4.9.1
scikit-learn==1.7.0
scipy==1.16.0
six==1.17.0
sniffio==1.3.1
soupsieve==2.7
SQLAlchemy==2.0.41
tenacity==9.1.2
tensorboard==2.15.2
tensorboard-data-server==0.7.2
tensorflow==2.15.0
tensorflow-estimator==2.15.0
tensorflow-io-gcs-filesystem==0.37.1
# tensorflow-macos==2.15.0   # <--- DO NOT install in Linux containers
termcolor==3.1.0
threadpoolctl==3.6.0
tqdm==4.67.1
typing-inspect==0.9.0
typing-inspection==0.4.1
typing_extensions==4.14.0
tzdata==2025.2
urllib3==2.5.0
websockets==15.0.1
Werkzeug==3.1.3
wrapt==1.14.1
yarl==1.20.1
yfinance==0.2.63
zstandard==0.23.0
EOF'
REQ

RUN python -m pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
WORKDIR /app
COPY . /app

# 로그/모델/데이터 디렉터리 미리 생성 (없어도 무방)
RUN mkdir -p /app/DataSets /app/SaveModel /app/LogFile

# TensorBoard 포트 (선택)
EXPOSE 6006

# 기본 실행 명령(예시): 필요에 따라 바꾸세요
# - 학습/예측 분기는 코드 내부에서 파일 존재 여부로 결정됩니다.
# CMD ["python", "main.py"]
