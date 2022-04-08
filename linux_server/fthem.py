from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import random
CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
                         options=chrome_options  #chrome_options=chrome_options
                         )
temp = random.randint(363, 365)/10
username="nusstu\\\E1234567"  #need 3 back slashes!
password="Passw0rd"
#driver = webdriver.Chrome('/usr/local/bin/chromedriver')
driver.get('https://myaces.nus.edu.sg/htd/');
html='''
document.getElementById(Login.userNameInput).value='{}'
document.getElementById(Login.passwordInput).value='{}'
Login.submitLoginRequest().value='true'
'''.format(username,password)
driver.execute_script(html)
javaScript='''
document.dlytemperature.symptomsFlag[0].checked='true'
document.dlytemperature.familySymptomsFlag[0].checked='true'
document.dlytemperature.temperature.value='{}'
document.dlytemperature.webdriverFlag.value ="N"
document.dlytemperature.submit()
'''.format(temp)
driver.execute_script(javaScript)
#print(driver.title)
driver.quit()
