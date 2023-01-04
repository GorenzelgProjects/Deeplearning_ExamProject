from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from numpy.linalg import norm
import nltk

import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd

from tkinter import *

from tkinter import *
from datetime import datetime
import re
from tkinter import messagebox
from tkinter.font import Font
import textwrap

import os
import openai

import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------------
OPEN_API_KEY="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"
openai.api_key = OPEN_API_KEY
#------------------------------------------------------------------------------------
#Loading in the ID's
model_checkpoint = "./bert_Finetuned_Squad_Local"
question_answerer = pipeline("question-answering", model=model_checkpoint)

#------------------------------------------------------------------------------------
model_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')

stopwords = nltk.corpus.stopwords.words('english')
# remove these words from stop words
stopword_exceptions = ['not']
 
# update the stopwords list without the words above
all_stopwords = [el for el in stopwords if el not in stopword_exceptions]

with open("./Amazon_data.txt") as f:
    sentences = f.readlines()

sentences_embeddings = model_embedding.encode(sentences)

short_term_embedding = 0
short_term_original = 0

memory_weight_1 = 0.40
memory_weight_2 = 0.60

q_weight = 0.10
a_weight = 0.90

#------------------------------------------------------------------------------------
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

item_categories = [
    'description', 
    'price',
    'rating',
    'stock',
    'discount',
    'saving']

search_tags = [
    "lowest",
    "highest",
    "lower than",
    "higher than"]

skip_tags = [
    "contextual"]

item_embeddings = model_embedding.encode(item_categories)

konf_pct = 0.92

bubbles = []
bubble_move = []

#------------------------------------------------------------------------------------
def remove_stopword(word):
    word_list = word.split()

    if len(word_list) > 1:
        temp_word = []
        for ele in word_list:
            if ele not in all_stopwords:
                temp_word.append(ele)
            word = ' '.join(temp_word)
            
    return word

def replace_word(word):
    word_list = word.split()

    if len(word_list) > 1:
        temp_word = []
        for ele in word_list:
            temp_word.append(ele.replace("cheapest","lowest").replace("cheaper","lower").replace("cheap","low"))
            word = ' '.join(temp_word)
            
    else:
        word = word.replace("cheapest","lowest").replace("cheaper","lower").replace("cheap","low")

    return word
#------------------------------------------------------------------------------------
#Key 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
def gpt_call(answer):  
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Please rewrite this sentence: {}".format(answer),
    temperature=0.7,
    max_tokens=709,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    
    return response.choices[0].text

#------------------------------------------------------------------------------------
##Finds the context from the two nearest dots confining our answer
def proper_context(context,question):
    
    answer = question_answerer(question=question, context=context)
    #Saving start and end ID of answer
    start = answer["start"]
    end = answer["end"]

    dots = []

    #Finds the dots in the senence
    for idx, letters in enumerate(context):
        if letters == "." and context[idx+1] == " ":
            dots.append(idx)
    #print(dots)
    start_context = []
    end_context = []
    
    #measuring the dot's distance from there start and end ID of answer
    for i in range(len(dots)):
        a = dots[i]-start
        b = dots[i]-end
        start_context.append(a)
        end_context.append(b)

    #Finding the closest dots on each side.
    #a = np.argmin(start_context)
    start_context = np.array(start_context)

    dots_before_index = np.where(start_context[:] <= 0)[0]
    dots_before_value = np.amax(start_context[dots_before_index])
    dots_before = np.where(start_context[:] == dots_before_value)[0]

    a = dots_before[0]
    

    end_context = np.array(end_context)

    dots_after_index = np.where(end_context[:] >= 0)[0]
    dots_after_value = np.amin(end_context[dots_after_index])
    dots_after = np.where(end_context[:] == dots_after_value)[0]

    b = dots_after[0]

    #dots.pop(a)
    idx_start = dots[a]
    idx_end = dots[b]
    #print(idx_end)
    done = np.sort(np.array([int(idx_start),int(idx_end)]))

    proper_context = context[done[0]+2:done[1]+1]

    return proper_context, answer


    
def context_search(question, question_embedding, short_team_embedding, first_question=True):
    compare_embedding_mean = []

    history_question = ''

    if first_question:       

        for sentence, embedding in zip(sentences, sentences_embeddings):
            compare_embedding_mean.append(np.mean(np.square(question_embedding - embedding)).mean())

        best_compare_index = np.argmin(compare_embedding_mean)
        best_sentence = sentences[best_compare_index]
        
        history_question += question + ' ' + best_sentence + ' '

        short_team_embedding = (question_embedding*q_weight) + (sentences_embeddings[best_compare_index]*a_weight)

        first_question = False

        print("Context:", best_sentence)
        print('________________________________________________')

    else:

        short_team_embedding = (short_team_embedding*memory_weight_1) + (question_embedding*memory_weight_2)
        
        for sentence, embedding in zip(sentences, sentences_embeddings):

            pair_CSM = np.dot(short_team_embedding,embedding)/(norm(short_team_embedding)*norm(embedding))

            compare_embedding_mean.append(pair_CSM)

        sorted_index = np.argsort(compare_embedding_mean).tolist()
        sorted_embeddings = []
        sorted_context = []
        for i in sorted_index:
            sorted_embeddings.append(compare_embedding_mean[i])
            sorted_context.append(sentences[i])

        best_compare_index = np.argmax(compare_embedding_mean)
        best_sentence = sentences[best_compare_index]
        history_question += ' ' + best_sentence + ' '

        short_team_embedding = (short_team_embedding*memory_weight_1) + (sentences_embeddings[best_compare_index]*memory_weight_2)

        print('------------------------------------------------')
        print("Context:", best_sentence)
        print('________________________________________________')

    return best_sentence, short_team_embedding, first_question
#------------------------------------------------------------------------------------
def general_chat(data_pd, question, question_short_embedding):
    bot_name = "Bot"
    match_index = 0
    answer = ''
    true_answer = False
    search_context = False
    q_to_a = False

    question = tokenize(question)
    X = bag_of_words(question, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    item_tag = ''
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > konf_pct:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                item_tag = tag
                if item_tag in skip_tags:
                    true_answer = False
                    search_context = True
                    q_to_a = True
                    return answer, true_answer, search_context, q_to_a, match_index

                elif item_tag not in search_tags and item_tag not in skip_tags:
                    answer = random.choice(intent['responses'])
                    true_answer = True
                    search_context = False
                    q_to_a = False
                    return answer, true_answer, search_context, q_to_a, match_index
                elif item_tag in search_tags and item_tag not in skip_tags:
                    compare_embedding_mean = []
                    for sentence, embedding in zip(item_categories, item_embeddings):
                        #print("Context:", sentence)
                        compare_embedding_mean.append(np.mean(np.square(question_short_embedding - embedding)).mean())

                    best_compare_index = np.argmin(compare_embedding_mean)
                    search_item = item_categories[best_compare_index]
                    print(item_tag, search_item)
                    answer, match_index = general_search(data_pd, search_type=item_tag, search_value=0, search_item=search_item)
                    
                    if item_tag == "lower than" or item_tag == "higher than":
                        pass

                    true_answer = False
                    search_context = False
                    q_to_a = True
                    return answer, true_answer, search_context, q_to_a, match_index
    else:
        true_answer = False
        search_context = True
        q_to_a = True
        return answer, true_answer, search_context, q_to_a, match_index

#------------------------------------------------------------------------------------
def data_to_panda(data):
    discount = []
    saving = []
    price = []
    stock = []
    sale = []
    rating = []
    for i in range(len(data)):
        temp_stock = str(data[i,3]).replace(u'\uff0c',',').replace(",","")
        discount.append((1-float(data[i,1][1:].replace(u'\uff0c',',').replace(",",""))/float(data[i,4][1:].replace(u'\uff0c',',').replace(",","")))*100)
        saving.append(float(data[i,4][1:].replace(u'\uff0c',',').replace(",",""))-float(data[i,1][1:].replace(u'\uff0c',',').replace(",","")))
        price.append(float(data[i,1][1:].replace(u'\uff0c',',').replace(",","")))
        sale.append(float(data[i,4][1:].replace(u'\uff0c',',').replace(",","")))
        stock.append(float(temp_stock))
        rating.append(float(data[i,2]))
    data_dict = {
        'description': data[:,0], 
        'price': price,
        'rating': rating,
        'stock': stock,
        'sale': sale,
        'discount': discount,
        'saving': saving
    }

    data_pd = pd.DataFrame(data_dict)

    return data_pd
#------------------------------------------------------------------------------------
def general_search(data, search_type='lowest', search_value=0, search_item='price'):
    result_idx_list = []
    if search_type == 'lowest':
        result_idx_list.append(np.argmin(data[search_item]))
    elif search_type == 'highest':
        result_idx_list.append(np.argmax(data[search_item]))
    elif search_type == 'lower than':
        result_idx_list = np.where(data[search_item] < search_value)[0]
    elif search_type == 'higher than':
        result_idx_list = np.where(data[search_item] > search_value)[0]
    elif search_type == 'equal to':
        result_idx_list = np.where(data[search_item] == search_value)[0]

    return sentences[result_idx_list[0]], result_idx_list[0]
#------------------------------------------------------------------------------------
class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        BG_GRAY = "#59B8FF"
        BG_COLOR = "#EEEEEE"
        BG_COLOR_2 = "#F9F9F9"
        TEXT_COLOR = "#000000"

        FONT = "Helvetica 10"
        FONT_BOLD = "Helvetica 12 bold"

        self.window.title('IMMABOT')
        self.window.resizable(width=False, height=False)
        self.window.configure(width=300, height=700, bg=BG_COLOR)

        #head label
        head_label = Label(self.window, bg=BG_GRAY, fg=TEXT_COLOR, text="Welcome to IMMABOT", 
                           font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        self.text_widget = Canvas(self.window, width=200, height=200,bg="white")
        self.text_widget.place(relheight=0.85, relwidth=1, rely=0.08)
        #self.text_widget.configure(cursor="arrow", state=DISABLED)
       

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=60)
        bottom_label.place(relwidth=1, rely=0.9)

        #message entry box
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR_2, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.04, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        self.frame = Frame(self.text_widget, bg='white')

        scrollbar = Scrollbar(self.frame, orient="vertical", command=self.text_widget.yview)


        self.frame.bind(
            "<Configure>",
            lambda e: self.text_widget.configure(
                scrollregion=self.frame.bbox("all")
            )
        )

        self.text_widget.create_window((0, 0), window=self.frame, anchor="nw")
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        # send button
        send_button = Button(bottom_label, text="Send", 
                             font=FONT_BOLD, width=20, 
                             bg=BG_COLOR_2, 
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.04, relwidth=0.22)

        scrollbar.place(relx=0.95, rely=0.1, relheight=0.7)

    def bot_bubble(self,master,x,y,color,choice,message=""):
        self.master = master
        self.frame = Frame(self.master, bg=color)
        #self.frame = self.scrollable_frame_1
        self.i1 = self.master.create_window(x,y, window=self.frame)       
        Label(self.frame,text=datetime.now().strftime("%d-%m-%Y %X"),font=("Helvetica", 7),bg=color).grid(row=0,column=0,sticky="w",padx=5) #tarih saat        
        Label(self.frame, text=textwrap.fill(message, 20), font=("Helvetica", 9),bg=color).grid(row=1, column=0,sticky="w",padx=5,pady=3)
        self.window.update_idletasks()

        if choice ==1:
            p1,p2,p3,p4,p5,p6,(x1,y1,x2,y2) = self.draw_triangle_1(self.i1)
            self.master.create_polygon((p1,p2,p3,p4,p5,p6), fill=color, outline=color)
        else:
            p1,p2,p3,p4,p5,p6,(x1,y1,x2,y2) = self.draw_triangle_2(self.i1)
            self.master.create_polygon((p1,p2,p3,p4,p5,p6), fill=color, outline=color)

        return (x1,y1,x2,y2)

    def draw_triangle_1(self,widget):
        x1, y1, x2, y2 = self.master.bbox(widget)
        return x1, y2 - 10, x1 - 15, y2 + 10, x1, y2, (x1, y1, x2, y2)

    def draw_triangle_2(self,widget):
        x1, y1, x2, y2 = self.master.bbox(widget)
        return x2, y2 - 10, x2 + 15, y2 + 10, x2, y2, (x1, y1, x2, y2)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        if msg == 'flush()':
            self.msg_entry.delete(0, END)
            self.text_widget.delete(ALL)
            global short_term_embedding
            global first_question

            short_term_embedding = 0
            first_question = True
        
        else:
            self.msg_entry.delete(0, END)
            msg1 = f"{sender}: {msg}\n\n"
            self.text_widget.configure(state=NORMAL)

            input_message = msg1

            if bubbles:      
                self.text_widget.move(ALL, 0, -(60+int((bubbles[-1][-1]-bubbles[-1][1])/2)))

            a = self.bot_bubble(self.text_widget,80,400, color="light green", choice=1, message=input_message)
            bubbles.append(a)

            print(bubbles[-1][-1]-bubbles[-1][1])

            answer = get_response(msg)
            msg2 = f"Bot: {answer}\n\n"
            output_message = msg2

            if bubbles:      
                self.text_widget.move(ALL, 0, -(60+int((bubbles[-1][-1]-bubbles[-1][1])/2)))
        
            b = self.bot_bubble(self.text_widget,200,420, color="light blue", choice=2, message=output_message)
            bubbles.append(b)
        

#------------------------------------------------------------------------------------
def get_response(question):

    if question == "":
        return "I'm sorry, I didn't get that."

    question_edit = question
    if question_edit[-1] in eos_tokens:
        question_edit = question_edit[:-1]
    question_edit += '?'

    question_short = remove_stopword(question)
    question_short = replace_word(question_short)
    question_short_embedding = model_embedding.encode(question_short)       
    question_embedding = model_embedding.encode(question_edit)
    global short_term_embedding
    global first_question

    print("Question: ",question_edit)
    print('--------------------------------------------------------')

    context, true_answer, search_context, q_to_a, match_index = general_chat(data_pd, question_edit, question_embedding)

    if true_answer:
        answer = context
        print("Answer: ", answer)
        print('--------------------------------------------------------')
    else:
        answer = ''

        if search_context:
            context, short_term_embedding, first_question = context_search(question_edit, question_embedding, short_term_embedding, first_question)
            q_to_a = True
        else:
            if first_question:
                short_term_embedding = question_embedding*q_weight + sentences_embeddings[match_index]*a_weight
            else:
                short_term_embedding = short_term_embedding*memory_weight_1 + (question_embedding*q_weight + sentences_embeddings[match_index]*a_weight)*memory_weight_2
            first_question = False
            print(context)

        if q_to_a:
            #answer = context_to_answer(question_edit, context)
            full_answer, answer_dict = proper_context(context, question_edit)
            #answer = answer_dict['answer']
            print(answer_dict)
            if full_answer != ' ':
                answer = full_answer
            else:
                answer = answer_dict['answer']

            if answer_dict['score'] <= 0.10 and answer_dict['score'] > 0.01:
                answer = "I'm a bit unsure what you're asking for, but I found this: " + answer
            elif answer_dict['score'] <= 0.01:
                answer = "I found this:"+ answer + " But if this isn't what you're looking for, please rephrase your question?"

    answer = gpt_call(answer)

    return answer


first_question = True
eos_tokens = [".",",","!","?",":",";"]

data = pd.read_csv("results_outlet_done.csv", sep='|', error_bad_lines=False)
data = np.array(data)

data_pd = data_to_panda(data)

app = ChatApplication()
app.run()
#------------------------------------------------------------------------------------