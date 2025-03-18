# note 필수 import part
from selenium import webdriver

# note 
# Keys = enter, ecs를 입력하고 싶을 때
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# todo 어느 부분에서 사용되는지 체크
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# note load env file and keyword 'os'
from dotenv import load_dotenv
import os

# note 편의성 부분 - copy file or dir
import pyperclip

# .env file load
load_dotenv()

# selenium high version에서는 경로를 지정해줄 필요가 없음
openWindow = webdriver.ChromeOptions()
openWindow.add_argument('https://www.financialjuice.com/home')

# todo : 좌측 실시간 지표 결과 크롤링


# info : 시간 / 중요도 / (나라 - 지표명) / 실제 값 / 예측 값 / 이전 값



# todo :메인화면 실시간 속보 결과 크롤링




# info 종료 방지
input('enter')