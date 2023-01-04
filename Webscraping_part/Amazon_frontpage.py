#Disclaimer the current code worked at use time, the cod4e may not work later beacuse of Amazon.com changing structur or add new symbols.

##This code webscrapes the Amazon front page and saves the product information in a .csv-file for further downstream usage.

import csv
from bs4 import BeautifulSoup
from selenium import webdriver

#This ensure that the scraper goes through all pages.
def get_url(search_term):
    template = 'https://www.amazon.com/s?{}=517808&qid=1669912541&ref=sr_pg_1'
    search_term = search_term.replace(' ','+')
    
    #add term query to url
    url = template.format(search_term)
    
    #add page
    url += '&page={}'
    
    return url

#Extract all the wanted data.
def extract_record(item):
    atag = item.h2.a
    #Titel/description
    description = atag.text.strip()
    try:
        #Price
        price_parent = item.find('span', 'a-price')
        price = price_parent.find('span', 'a-offscreen').text
        #print(price)
    except AttributeError:
        return
    
    try:
        #Rating
        rating_string = item.i.text
        list_rating= list(rating_string.split(" "))
        rating = list_rating[0]
        
        #Stock
        stock = item.find('span',{'class': 'a-size-base s-underline-text'}).text
        
    except AttributeError:
        rating = ''
        stock = ''
    
    try:
        #Before pirce
        sale_parent = item.find('span', 'a-price a-text-price')
        sale = sale_parent.find('span', 'a-offscreen').text
    except AttributeError:
        sale = price
        
    result = (description,price,rating,stock,sale)
    return result

#Take the link into the program.
def main(search_term):
    driver = webdriver.Chrome()
        
    records = []  
    url = get_url(search_term)    
    for page in range(1,7):
        driver.get(url.format(page))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results = soup.find_all('div',{'data-component-type': 's-search-result'})
                
        for item in results:
            record = extract_record(item)
            if record:
                        records.append(record)
           
    driver.close()
             
    #Save in csv
    with open('results_outlet_done.csv','w',newline = '', encoding = 'utf-8') as f:
        writer = csv.writer(f, delimiter = '|')
        writer.writerow(['Description','Price','Rating','In_Stock','Sale'])
        writer.writerows(records)
#The link that we used.  
main('https://www.amazon.com/s?srs=517808&qid=1669912541&ref=sr_pg_1')