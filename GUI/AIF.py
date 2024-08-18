
import random
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import filedialog
from datetime import datetime

import os
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
# from sklearn import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score)
import joblib
from sklearn.model_selection import GridSearchCV


class PublicMod(Tk):
    def __init__(self):
        super().__init__()

    def logo(self,parent):
        image = Image.open('./Logo.png').resize((100,60))
        global photo
        photo = ImageTk.PhotoImage(image)
        # photo = PhotoImage(file='./att.png')
        logo = Label(parent,image=photo,text='Test')
        logo.place(x=850, y=20, width=100, height=60)
        # logo.pack()
        return logo

    def citation(self,parent):
        citation = Label(parent,text='Reference:\n[1].Huang, Q., Zhang, H., Zhang, L., and Xu, B. (2023). Bacterial microbiota in different types of processed meat products: diversity, adaptation, and co-occurrence. Critical Reviews in Food Science and Nutrition. 1-16.',anchor=NW,wraplength=800)
        citation.place(x=50,y=470,width=900,height=90)
        return citation

    def copyright(self,parent):
        copyright = Label(parent,text='Copyright@2024')
        copyright.place(x=440,y=550,width=120,height=50)
        return copyright

    def close(self):
        self.destroy()

    def button1(self,parent,fg='black'):
        btn = tk.Button(parent, text='Home',font=('Time New Roman',18,'bold'), bd=5, relief=RAISED,fg=fg, takefocus=False, command=self.toMain)
        btn.place(x=50, y=20, width=120, height=60)
        return btn

    def button2(self,parent,fg='black'):
        btn = tk.Button(parent, text="Predict",font=('Time New Roman',18,'bold'), bd=5,relief=RAISED,fg=fg, takefocus=False, command=self.toPredict)
        btn.place(x=200, y=20, width=120, height=60)
        return btn

    def button3(self,parent,fg='black'):
        btn = tk.Button(parent, text="Retrain",font=('Time New Roman',18,'bold'), bd=5, relief=RAISED,fg=fg,takefocus=False, command=self.toRetain)
        btn.place(x=350, y=20, width=120, height=60)
        return btn

    def toMain(self):
        self.close()
        Main()

    def toPredict(self):
        self.close()
        Predict()

    def toRetain(self):
        self.close()
        Retrain()

class Main(PublicMod):
    def __init__(self):
        super().__init__()
        # self.__init__()
        self.initPage()

    def initPage(self):
        self.__win()
        self.logo = PublicMod.logo(self,self)
        self.bt1 = PublicMod.button1(self,self,fg='blue')
        self.bt2 = PublicMod.button2(self,self)
        self.bt3 = PublicMod.button3(self,self)

        self.tk_canvas_luwngo57 = self.__tk_canvas_luwngo57(self)
        self.tk_canvas_luwngvyq = self.__tk_canvas_luwngvyq(self)
        self.show_pre1 = self.show_pre(self)
        self.show_tra1 = self.show_tra(self)
        self.tk_button_luwnhi2a = self.__tk_button_luwnhi2a(self)
        self.tk_button_luwnhp9d = self.__tk_button_luwnhp9d(self)

        self.prelabeltxt = self.prelabel(self)
        self.trainlabeltxt = self.tralabel(self)

        self.citation = PublicMod.citation(self, self)
        self.copyright = PublicMod.copyright(self, self)

    def __win(self):
        self.title("AIF")
        # 设置窗口大小、居中
        width = 1000
        height = 600
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def show_pre(self,parent):
        imagePre = Image.open('./Pre1.png').resize((260, 260))
        global pre
        pre = ImageTk.PhotoImage(imagePre)
        # photo = PhotoImage(file='./att.png')
        prefig = Label(parent, image=pre,text='Predict')
        prefig.place(x=130, y=150, width=260, height=260)
        return prefig

    def show_tra(self,parent):
        imageTra = Image.open('./Tra.png').resize((260, 260))
        global tra
        tra = ImageTk.PhotoImage(imageTra)
        # photo = PhotoImage(file='./att.png')
        trafig = Label(parent, image=tra,text='Train')
        trafig.place(x=610, y=150, width=260, height=260)
        return trafig

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def prelabel(self,parent):
        citation = tk.Label(parent,text='Predicting with pre-trained model',font=('Time New Roman',16,'bold'),wraplength=800)
        citation.place(x=60,y=105,width=400,height=40)
        return citation

    def tralabel(self,parent):
        citation = tk.Label(parent,text='Training Model with input data',font=('Time New Roman',16,'bold'),wraplength=800)
        citation.place(x=540,y=105,width=400,height=40)
        return citation

    def __tk_canvas_luwngo57(self, parent):
        canvas = tk.Canvas(parent, bg="#F5FFFA")
        canvas.place(x=50, y=100, width=420, height=360)
        return canvas

    def __tk_canvas_luwngvyq(self, parent):
        canvas = tk.Canvas(parent, bg="#F5FFFA")
        canvas.place(x=530, y=100, width=420, height=360)
        return canvas

    def __tk_button_luwnhi2a(self, parent):
        btn = tk.Button(parent, text="Start", font=('Time New Roman',18,'bold'),takefocus=False, command=self.changePredict)
        btn.place(x=210, y=415, width=100, height=40)
        return btn

    def __tk_button_luwnhp9d(self, parent):
        btn = tk.Button(parent, text="Start", font=('Time New Roman',18,'bold'),takefocus=False, command=self.changeRetrain)
        btn.place(x=690, y=415, width=100, height=40)
        return btn

    def close(self):
        self.destroy()

    def changeRetrain(self):
        self.close()
        Retrain()

    def changePredict(self):
        # self.initPage.destroy()
        self.close()
        Predict()


class Predict(PublicMod):
    modelname = './best_random_forest_model.pkl'
    def __init__(self):
        super().__init__()
        self.initPage()

    def initPage(self):
        self.__win()
        # self.logo = self.__logo(self)

        self.logo = PublicMod.logo(self, self)
        self.bt1 = PublicMod.button1(self, self)
        self.bt2 = PublicMod.button2(self, self,fg='blue')
        self.bt3 = PublicMod.button3(self, self)

        self.tk_canvas_luwn9pcs = self.__tk_canvas_luwn9pcs(self)
        self.label1 = self.__tk_select_label_0(self)
        self.tk_select_box_1 = self.__tk_select_box_luwn89r8(self)
        self.label2 = self.__tk_select_label_1(self)
        self.tk_select_box_2 = self.__tk_select_box_luwn96sy(self)
        self.label3 = self.__tk_select_label_2(self)
        self.tk_select_box_3 = self.__tk_select_box_luwn9adf(self)
        self.label4 = self.__tk_select_label_3(self)
        self.tk_select_box_4 = self.__tk_select_box_luwn9jnv(self)
        self.tk_canvas_luwna90y = self.__tk_canvas_luwna90y(self)
        # self.tk_canvas_luwncamy = self.__tk_canvas_luwncamy(self)
        # self.tk_canvas_luwncnrm = self.__tk_canvas_luwncnrm(self)

        self.selectBt = self.buttonSelect(self)

        self.tk_predict_button = self.__tk_button_luwnhp9d(self)

        self.citation = PublicMod.citation(self, self)
        self.copyright = PublicMod.copyright(self, self)


    def __win(self):
        self.title("AIF")
        # 设置窗口大小、居中
        width = 1000
        height = 600
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        
        self.resizable(width=False, height=False)

    def close(self):
        self.destroy()

    def select_file(self):
        filename = filedialog.askopenfilename(title="Open file")
        self.modelname = filename
        self.show_prediction(self,filename)
        # if filename != '':
        #     content = read_

    def buttonSelect(self,parent):
        bt = tk.Button(parent,text='InputModel',font=('Time New Roman',18,'bold'),command=self.select_file)
        bt.place(x=200,y=160,width=150,height=30)

    # def getSelectValue(self):

    def scrollbar_autohide(self,vbar, hbar, widget):
        """自动隐藏滚动条"""
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)
        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)
        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())
    
    def v_scrollbar(self,vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self,hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self,master, widget,is_vbar,is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_canvas_luwn9pcs(self,parent):
        canvas = Canvas(parent,bg="#F5FFFA")
        canvas.place(x=50, y=140, width=420, height=360)
        return canvas

    def __tk_canvas_luwna90y(self,parent):
        canvas = Canvas(parent,bg="#F5FFFA")
        canvas.place(x=530, y=140, width=420, height=360)
        return canvas
    ### selection
    def __tk_select_label_0(self,parent):
        label = Label(parent,text='Packing Method')
        label.place(x=80,y=210,width=150,height=30)
        return label

    def __tk_select_box_luwn89r8(self,parent):
        cb = Combobox(parent, state="readonly",)
        cb['values'] = ("Plain Packaging","Vacuum Packaging","Modified Atmosphere Packaging")
        cb.current(0)
        cb.place(x=230, y=210, width=220, height=30)
        return cb

    def __tk_select_label_2(self, parent):
        label = Label(parent, text='Preservatives')
        label.place(x=80, y=250, width=150, height=30)
        return label

    def __tk_select_box_luwn9adf(self, parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("None", "nisin", "ε-polylysine", 'Chitosan', 'Composite Preservatives')
        cb.current(0)
        cb.place(x=230, y=250, width=220, height=30)
        return cb

    def __tk_select_label_1(self,parent):
        label = Label(parent,text='Storage temperature')
        label.place(x=80,y=290,width=150,height=30)
        return label

    def __tk_select_box_luwn96sy(self,parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("Low-temperature（0~10℃)","Ambient Temperature(15-25℃)",">25℃")
        cb.current(0)
        cb.place(x=230, y=290, width=220, height=30)
        return cb

    def __tk_select_label_3(self,parent):
        label = Label(parent,text='Secondary sterilization')
        label.place(x=80,y=330,width=150,height=30)
        return label
    def __tk_select_box_luwn9jnv(self,parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("None","Low-Temperature Sterilization","Mid-High-Temperature Sterilization",'Microwave Sterilization')
        cb.current(0)
        cb.place(x=230, y=330, width=220, height=30)
        return cb

    def __tk_button_luwnhp9d(self, parent):
        btn = tk.Button(parent, text="Predict", font=('Time New Roman',18,'bold'),takefocus=False, command=self.getValues)
        btn.place(x=200, y=400, width=150, height=30)
        return btn

    def show_prediction(self,parent,result,data):
        # text1 = 'Model:' + self.modelname + '\nResult:' + str(result) + '\n'
        text1 = ''
        text2 = ''
        if data['package'] == 1:
            text2 += 'Vacuum packaging or Modified Atmosphere Packaging  may improve shelf life.'
        if data['sterilization'] == 1:
            text2 += 'Potential sterilization method，such as low temperature sterilization, medium and high temperature sterilization or microwave sterilization may improve the shelf life.'
        if data['temperature'] == 1:
            text2 += 'In this case, lowering storage temperature may be an effective strategy to extend shelf life.'
        if data['preservative'] == 1:
            text2 += 'Shelf life may be extended by adding preservatives.\n'
        # for kw in data.keys():
        #     text2 += kw +':' + str(data[kw])+'\n'
        lb = tk.Label(parent,text=text1+text2,font=('Times New Roman',12,'bold'),fg='teal',anchor='w',width=32,wraplength=320,justify='left')
        lb.place(x=560, y=240, width=350, height=220)
        return lb

    def show_results(self,parent,result):
        if result > 5 or result < 1:
            result = 0
        imageR = Image.open('./L{result}.png'.format(result=result)).resize((300, 80))
        # image = Image.open('./Logo.png'.format(result=result)).resize((100, 100))
        global p1
        p1 = ImageTk.PhotoImage(imageR)
        # photo = PhotoImage(file='./att.png')
        resultlabel = Label(parent, image=p1,text='Result')
        resultlabel.place(x=580, y=150, width=300, height=80)
        return resultlabel


    def getValues(self):
        packagelists = ["Plain Packaging","Vacuum Packaging","Modified Atmosphere Packaging"]
        temperaturelists = ["Low-temperature（0~10℃)","Ambient Temperature(15-25℃)",">25℃"]
        preservativelists = ["None","nisin","ε-polylysine",'Chitosan','Composite Preservatives']
        sterilizationlists = ["None","Low-Temperature Sterilization","Mid-High-Temperature Sterilization",'Microwave Sterilization']
        value1 = packagelists.index(self.tk_select_box_1.get()) + 1 #package
        value2 = temperaturelists.index(self.tk_select_box_2.get()) + 1 #temperature
        value3 = preservativelists.index(self.tk_select_box_3.get()) + 1 #preservative
        value4 = sterilizationlists.index(self.tk_select_box_4.get()) + 1 #sterilization
        data = {'package':value1,'temperature':value2,'preservative':value3,'sterilization':value4}
        result = predict(self.modelname,data)
        self.show_prediction(self,result,data)
        self.show_results(self,result)


class Retrain(PublicMod):
    def __init__(self):
        super().__init__()
        self.initPage()

    def initPage(self):
        self.__win()

        self.logo = PublicMod.logo(self, self)
        self.bt1 = PublicMod.button1(self, self)
        self.bt2 = PublicMod.button2(self, self)
        self.bt3 = PublicMod.button3(self, self,fg='blue')

        self.tk_canvas_luwngo57 = self.__tk_canvas_luwngo57(self)
        # self.tk_canvas_luwngvyq = self.__tk_canvas_luwngvyq(self)
        self.input = self.input(self)
        self.submitbutton = self.submitBotton(self)

        self.citation = PublicMod.citation(self, self)
        self.copyright = PublicMod.copyright(self, self)

    def __win(self):
        self.title("AIF")
        # 设置窗口大小、居中
        width = 1000
        height = 600
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.resizable(width=False, height=False)

    def __tk_canvas_luwngo57(self, parent):
        canvas = Canvas(parent, bg="#F5FFFA")
        canvas.place(x=50, y=100, width=900, height=270)
        return canvas

    def __tk_canvas_luwngvyq(self, parent):
        canvas = Canvas(parent, bg="#F5FFFA")
        canvas.place(x=430, y=100, width=390, height=300)
        return canvas

    def close(self):
        self.destroy()

    def input(self,parent):
        text = '''temperature,package,preservative,sterilization,target
1,1,1,1,1
1,1,4,1,1
1,3,1,1,1'''
        # entry = tk.Entry(parent,font=('Times New Roman',12))
        # entry.insert(0,text)
        entry = tk.Text(parent,wrap=WORD)
        entry.insert('0.0',text)
        entry.place(x=50,y=100,width=900,height=270)
        return entry

    def submitBotton(self,parent):
        bt = tk.Button(parent,text="submit",font=('Times New Roman',18,'bold'),command=self.getValues)
        bt.place(x=450,y=380,width=100,height=30)
        return bt

    def getValues(self):
        values = self.input.get('1.0','end')
        # print(values)
        modelname = datetime.strftime(datetime.now(),'%Y-%m%d-%H-%M-%S') + '.pkl'
        X,Y = [],[]
        for line in values.split('\n')[1:]:
            tmp_x = []
            if len(line.split(',')) != 5:
                continue
            for item in line.split(','):
                print(item)
                tmp_x.append(float(item))
            X.append(tmp_x[:4])
            Y.append(tmp_x[-1])
        if len(X) > 2:
            state = training(X,Y,modelname)
        else:
            state = -1
            modelname = 'None'

        self.show_prediction(self,state,modelname)


    def show_prediction(self,parent,state,modelname):
        dirpath = os.getcwd()
        text1 = 'successful'
        text2 = 'unsuccessful'
        if state == 1:
            lb = tk.Label(parent,text='Model:' + dirpath + '/' + modelname + ', Traing ' + text1,font=('Times New Roman',12,'bold'))
        else:
            lb = tk.Label(parent,text='Model: None, Training ' + text2,font=('Times New Roman',18,'bold'))
        lb.place(x=100, y=420, width=800, height=30)
        return lb


class Win(Main):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)
    def __event_bind(self):
        pass
    def __style_config(self):
        pass


def training(X_train,y_train,modelname):
    data2 = pd.read_csv("data22.csv")
    traindatax = []
    for data in X_train:
        temperature,package,preservative,sterilization = data

        # matching_rows = data2[(data2['temperature'] == temperature) &
        #                       (data2['package'] == package) &
        #                       (data2['preservative'] == preservative) &
        #                       (data2['sterilization'] == sterilization)]


        mean_target_temperature = data2[(data2['temperature'] == temperature)]['target'].mean()
        mean_target_package = data2[(data2['package'] == package)]['target'].mean()
        mean_target_preservative = data2[(data2['preservative'] == preservative)]['target'].mean()
        mean_target_sterilization = data2[(data2['sterilization'] == sterilization)]['target'].mean()

        traindatax.append([mean_target_temperature, mean_target_package, mean_target_preservative, mean_target_sterilization])

    # inputdata = {'temperature': [mean_target_temperature], 'package': [mean_target_package],
    #              'preservative': [mean_target_preservative], 'sterilization': [mean_target_sterilization]}
    #
    # inputdata1 = [[mean_target_temperature, mean_target_package, mean_target_preservative, mean_target_sterilization]]

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    try:
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(traindatax, y_train)

        # Get the best model from grid search
        best_rf_model = grid_search.best_estimator_

        # Save the best model
        best_model_filename = modelname
        joblib.dump(best_rf_model, best_model_filename)
        print(f"Best RandomForestClassifier model saved as '{best_model_filename}'")
        return 1
    except:
        return -1

def predict(modelname,newdata):
    try:
        data2 = pd.read_csv("data22.csv")
        temperature = newdata['temperature']
        package = newdata['package']
        preservative = newdata['preservative']
        sterilization = newdata['sterilization']
        # matching_rows = data2[(data2['temperature'] == temperature) &
        #                       (data2['package'] == package) &
        #                       (data2['preservative'] == preservative) &
        #                       (data2['sterilization'] == sterilization)]

        mean_target_temperature = data2[(data2['temperature'] == temperature)]['target'].mean()
        mean_target_package = data2[(data2['package'] == package)]['target'].mean()
        mean_target_preservative = data2[(data2['preservative'] == preservative)]['target'].mean()
        mean_target_sterilization = data2[(data2['sterilization'] == sterilization)]['target'].mean()

        inputdata = {'temperature':[mean_target_temperature],'package':[mean_target_package],
                     'preservative':[mean_target_preservative],'sterilization':[mean_target_sterilization]}

        inputdata1 = [[mean_target_temperature,mean_target_package,mean_target_preservative,mean_target_sterilization]]

        model = joblib.load(modelname)
        # print(inputdata)

        # Use the modified newdata to make predictions
        predictions = model.predict(inputdata1)

        # Print the predictions
        # print(predictions)
        return predictions[0]
    except:
        return -1


if __name__ == "__main__":
    # root = Tk()
    # logo = PhotoImage(file='./logo1.png')
    # label = Label(root,image=logo)
    # label.place(x=38, y=432, width=795, height=200)
    # root.mainloop()
    win = Main()
    win.mainloop()