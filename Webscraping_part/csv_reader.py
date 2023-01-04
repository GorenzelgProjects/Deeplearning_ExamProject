#Disclaimer the current code worked at use time, the code may not work later beacuse of Amazon.com changing structur or add new symbols.

##This codes reads the CSV from "Amazon_frontpage.py" and creates a sentence for the model with the correct data and sends it to a txt.file

import pandas as pd
import numpy as np

#Open the csv file
data = pd.read_csv("results_outlet_done.csv", sep="|", error_bad_lines=False)
data = np.array(data)

description = data[:,0]
price = data[:,1]
rating = data[:,2]
stock = data[:,3]
sale = data[:,4]

#Ensure that the lenght is the same.
assert len(description) == len(price) == len(rating) == len(stock) == len(sale), "not same size"

f = open('Amazon_data.txt','a')
 
#Add all the different values to the text document.
for i in range(len(description)):
    item_rating = rating[i]
    item_description = description[i]
    item_before_price = sale[i]
    item_before_price = item_before_price[1:]
    item_before_price = item_before_price.replace(u'\uff0c',',').replace(",","")
    item_stock = str(stock[i])
    item_stock = item_stock.replace('(','').replace(')','')
    item_value = price[i]
    item_value = item_value[1:]
    item_value = item_value.replace(u'\uff0c',',').replace(",","")
    item_valuta = "$"
    item_price = item_valuta + item_value
    item_discount = (1 - float(item_value) / float(item_before_price)) * 100
    item_discount = round(item_discount,2)
    item_saving = float(item_before_price) - float(item_value)
    item_saving = round(item_saving,2)
    #item_url = url[i]
    #! The url for the product is: {}
    #, item_url
    context = """. The description of the chosen item is {}. The price of the product is {}. There´s currently {} in stock of the chosen product. You currently making a saving of {} compared to the usual price. This means you´re getting a percentural discount of {} percent. The user rating of the product is {} of 5 stars.""".format(item_description,item_price, item_stock, item_saving, item_discount, item_rating)
    context = context.replace(u'\uff0c',',').replace(u'\uff0f','/')

    
    f.write(context + ' \n')
f.close()

    