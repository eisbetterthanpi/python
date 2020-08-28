import webbrowser
from selenium import webdriver
import random

temp = random.randint(363, 365)/10
username="nusstu\\\E1234567"
password="password"

driver = webdriver.Chrome('C:\\Users\\hi\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\selenium\\webdriver\\chromedriver.exe')
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
# print(javaScript)

driver.execute_script(javaScript)
driver.quit() #comment this out to check if submission is successful
