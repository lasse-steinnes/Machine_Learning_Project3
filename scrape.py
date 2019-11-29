import pandas as pd 
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait

def scrape_data(data, driver):
    for row in driver.find_elements_by_xpath("//tbody"):
        temp = [td.text for td in row.find_elements_by_xpath(".//th[@class='result']")]
        if len(temp) !=0:
            if temp[0] =="num":
                #find labeles
                keyes = temp
                break

    for row in driver.find_elements_by_tag_name("tr"):
        elements = row.find_elements_by_tag_name("td")
        temp = [td.text for td in elements[slice(0,len(elements)-1)]]#ommit refno
        if len(temp) !=0:
            if temp[0].isdigit(): 
                """
                if any(data["num"]==temp[0]):
                    place = data["num"]==temp[0]
                    for key, value in zip(keyes,temp):
                        if any(key==data.columns):#skip existing
                            continue
                        else:
                            data.loc[place, key] = value
                else:
                """
                data = data.append({key:value for key, value in zip(keyes,temp)}, ignore_index=True)
    return data 

def condition(driver):
    nums = driver.find_elements_by_class_name("dataCount2")
    print(nums[-2].text, " of ", nums[-1].text)
    return nums[-1].text != nums[-2].text

#from BeautifulSoup  import BeautifulSoup

username = "lukasmat@student.matnat.uio.no"
password = "machinelearning"
search_page ="https://supercon.nims.go.jp/supercon/material_menu"

data = pd.read_csv("raw.csv")
#data = data.drop(columns=["figjc",	"hirfig", "hallfig", "thcfig", "tpfig"])
data_temp = pd.DataFrame()

opt = Options()
opt.add_argument("--headless")
driver = webdriver.Firefox(executable_path='/usr/bin/geckodriver', options=opt)
print("Startet Firefox")

driver.get(search_page)
driver.find_element_by_name("username").send_keys(username)
driver.find_element_by_name("password").send_keys(password)
driver.find_element_by_name("login").click()
print("LogIn successfull")

for i in range(15,18):
    print("Do option %i" %i)
    search_table = driver.find_element_by_name("material_search")
    Select(search_table.find_element_by_name("property")).select_by_value(str(i))
    search_table.find_element_by_xpath("//input[@type='submit']").click()
    print("Searched for option %i" %i)

    i = 1
    stop = 12
    off = 0
    while condition(driver):
        while i < stop:
            avalibale_pages = driver.find_elements_by_class_name("next")
            if len(avalibale_pages) < stop:
                stop = len(avalibale_pages)
            if i == stop -1 :
                avalibale_pages[i].click()
                i += 1
                continue             
            data_temp =scrape_data(data_temp, driver)
            print("Extracted page %i" % (i - off))
            avalibale_pages[i].click() 
            i += 1
            
        #dont load current page after next twice
        i = 2
        stop = 12
        off = 1
    data = data.append(data_temp)
    data.to_csv("raw.csv")  
    driver.get(search_page)
    


