import random
import tkinter as tk
from tkinter.ttk import Combobox
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import cycle
import joblib
import io
import webbrowser

# === 机器学习库 ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import category_encoders as ce

# === 全局配置 (字体与颜色) ===
config = {
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
}
rcParams.update(config)

# UI 样式配置
FONT_TITLE = ('Times New Roman', 24, 'bold')
FONT_BTN = ('Times New Roman', 16, 'bold')
FONT_TEXT = ('Times New Roman', 12)
FONT_LABEL = ('Times New Roman', 14, 'bold')
# 专门为Retrain输入框和Example定义的字体
FONT_INPUT = ('Times New Roman', 14) 
FONT_EXAMPLE = ('Times New Roman', 16, 'bold') # 错误弹窗中Example的字体

COLOR_HOME_BTN = "#E0F7FA"
COLOR_PRED_BTN = "#E8F5E9"
COLOR_TRAIN_BTN = "#FFF3E0"
COLOR_HIGHLIGHT = "#4CAF50"
COLOR_SUBMIT = "#2196F3"
COLOR_UPLOAD = "#FF9800"

# 数据校验规则
VALID_RANGES = {
    'temperature': [1, 2, 3],
    'package': [1, 2, 3],
    'preservative': [1, 2, 3, 4, 5],
    'sterilization': [1, 2, 3, 4],
    'target': [1, 2, 3, 4, 5]
}
REQUIRED_COLUMNS = ['temperature', 'package', 'preservative', 'sterilization', 'target']

# 文件路径
DATA_FILE_RAW = 'data22.csv'
DATA_FILE_ENCODED = 'leaveoneout_encoded.csv'
MODEL_FILE = 'best_random_forest_model.pkl'

# === 辅助功能函数 ===

def ensure_encoded_file_exists():
    if os.path.exists(DATA_FILE_ENCODED):
        return True
    if not os.path.exists(DATA_FILE_RAW):
        return False
    try:
        df_raw = pd.read_csv(DATA_FILE_RAW)
        encoder = ce.LeaveOneOutEncoder(cols=['temperature', 'package', 'preservative', 'sterilization'])
        encoded_df = encoder.fit_transform(df_raw.drop('target', axis=1), df_raw['target'])
        encoded_df['target'] = df_raw['target']
        encoded_df.to_csv(DATA_FILE_ENCODED, index=False)
        return True
    except Exception as e:
        print(f"Auto-generation failed: {e}")
        return False

def open_file(path):
    try:
        os.startfile(path)
    except AttributeError:
        import subprocess
        subprocess.call(['open', path])

# === GUI 类定义 ===

class PublicMod(tk.Tk):
    def __init__(self):
        super().__init__()

    def logo_section(self, parent):
        title_lbl = tk.Label(parent, text="AIF: Artificial Intelligence for Food", 
                             font=('Times New Roman', 20, 'bold', 'italic'), fg="#333333")
        title_lbl.place(x=50, y=30, height=40)
        try:
            pil_image = Image.open('./Logo.png')
            pil_image = pil_image.resize((100, 60), Image.Resampling.LANCZOS)
            global photo
            photo = ImageTk.PhotoImage(pil_image)
            logo = tk.Label(parent, image=photo)
            logo.place(x=850, y=20, width=100, height=60)
        except Exception:
            pass

    def citation(self, parent):
        citation = tk.Label(parent,
                         text='Reference:\n[1].Huang, Q., Zhang, H., Zhang, L., and Xu, B. (2023). Bacterial microbiota in different types of processed meat products: diversity, adaptation, and co-occurrence. Critical Reviews in Food Science and Nutrition. 1-16.',
                         font=('Times New Roman', 10), anchor="nw", justify="left", wraplength=900, fg="#555555")
        citation.place(x=50, y=520, width=900, height=60)
        return citation

    def copyright(self, parent):
        copy_lbl = tk.Label(parent, text='Copyright@2026', font=('Times New Roman', 10), fg="#888888")
        copy_lbl.place(x=440, y=570, width=120, height=20)
        return copy_lbl

    def close(self):
        self.destroy()

    def nav_buttons(self, parent, active_idx=0):
        common_opts = {'font': FONT_BTN, 'bd': 3, 'relief': 'raised', 'takefocus': False}
        c1 = "#BBDEFB" if active_idx == 1 else COLOR_HOME_BTN  
        c2 = "#C8E6C9" if active_idx == 2 else COLOR_PRED_BTN  
        c3 = "#FFE0B2" if active_idx == 3 else COLOR_TRAIN_BTN 

        btn1 = tk.Button(parent, text='Home', bg=c1, command=self.toMain, **common_opts)
        btn1.place(x=50, y=90, width=140, height=50)

        btn2 = tk.Button(parent, text="Predict", bg=c2, command=self.toPredict, **common_opts)
        btn2.place(x=210, y=90, width=140, height=50)

        btn3 = tk.Button(parent, text="Retrain", bg=c3, command=self.toRetain, **common_opts)
        btn3.place(x=370, y=90, width=140, height=50)

    def toMain(self):
        self.close()
        Main()

    def toPredict(self):
        self.close()
        Predict()

    def toRetain(self):
        self.close()
        Retrain()

    def base_window_setup(self):
        self.title("AIF - Artificial Intelligence for Food")
        width = 1000
        height = 600
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)
        self.resizable(False, False)


class Main(PublicMod):
    def __init__(self):
        super().__init__()
        self.base_window_setup()
        self.initPage()

    def initPage(self):
        self.logo_section(self)
        self.nav_buttons(self, active_idx=1) 
        self.show_img_section()
        self.citation(self)
        self.copyright(self)

    def show_img_section(self):
        try:
            img_p = Image.open('./Pre1.png').resize((240, 240), Image.Resampling.LANCZOS)
            self.photo_p = ImageTk.PhotoImage(img_p)
            lbl_p = tk.Label(self, image=self.photo_p, bg="white", bd=2, relief="groove")
            lbl_p.place(x=150, y=180, width=240, height=240)
            btn_p = tk.Button(self, text="Start Predict", font=FONT_BTN, bg=COLOR_PRED_BTN, command=self.toPredict)
            btn_p.place(x=190, y=440, width=160, height=40)
        except: pass

        try:
            img_t = Image.open('./Tra.png').resize((240, 240), Image.Resampling.LANCZOS)
            self.photo_t = ImageTk.PhotoImage(img_t)
            lbl_t = tk.Label(self, image=self.photo_t, bg="white", bd=2, relief="groove")
            lbl_t.place(x=610, y=180, width=240, height=240)
            btn_t = tk.Button(self, text="Start Retrain", font=FONT_BTN, bg=COLOR_TRAIN_BTN, command=self.toRetain)
            btn_t.place(x=650, y=440, width=160, height=40)
        except: pass
        
        tk.Label(self, text="Predict Shelf-life", font=FONT_LABEL).place(x=150, y=150, width=240)
        tk.Label(self, text="Retrain Model", font=FONT_LABEL).place(x=610, y=150, width=240)


class Predict(PublicMod):
    modelname = MODEL_FILE
    def __init__(self):
        super().__init__()
        self.base_window_setup()
        self.initPage()

    def initPage(self):
        self.logo_section(self)
        self.nav_buttons(self, active_idx=2) 

        tk.Canvas(self, bg="#F5F5F5", bd=0, highlightthickness=0).place(x=50, y=160, width=440, height=340)
        tk.Label(self, text="Input Model Parameters", font=('Times New Roman', 18, 'bold'), bg="#F5F5F5").place(x=120, y=170)

        self.create_comboboxes()

        tk.Button(self, text='Load Custom Model', font=('Times New Roman', 12), bg="#E0E0E0", 
                  command=self.select_file).place(x=190, y=220, width=160, height=30)

        tk.Button(self, text="PREDICT", font=('Times New Roman', 16, 'bold'), bg=COLOR_HIGHLIGHT, fg="white",
                  command=self.getValues).place(x=180, y=440, width=180, height=45)

        tk.Canvas(self, bg="#F0F4C3", bd=0, highlightthickness=0).place(x=510, y=160, width=440, height=340)
        
        self.result_text = tk.Text(self, font=('Times New Roman', 13), bg="#F0F4C3", bd=0, wrap="word")
        self.result_text.place(x=530, y=280, width=400, height=200)
        self.result_text.tag_configure("header", font=('Times New Roman', 14, 'bold'), foreground="#2E7D32")
        self.result_text.tag_configure("bullet", lmargin1=20, lmargin2=20, spacing1=5)
        
        self.result_text.insert("1.0", "Ready to predict...\nPlease select parameters on the left.")
        self.result_text.configure(state='disabled')

        self.citation(self)
        self.copyright(self)

    def create_comboboxes(self):
        labels = ['Packing Method', 'Preservatives', 'Storage Temperature', 'Secondary Sterilization']
        options = [
            ("Plain Packaging", "Vacuum Packaging", "Modified Atmosphere Packaging"),
            ("None", "nisin", "ε-polylysine", 'Chitosan', 'Composite Preservatives'),
            ("Low-temperature (0~10℃)", "Ambient Temperature (15-25℃)", ">25℃"),
            ("None", "Low-Temperature Sterilization", "Mid-High-Temperature Sterilization", 'Microwave Sterilization')
        ]
        self.combos = []
        start_y = 270
        for i, (lbl, opts) in enumerate(zip(labels, options)):
            y_pos = start_y + i * 40
            tk.Label(self, text=lbl, font=('Times New Roman', 12), bg="#F5F5F5", anchor='e').place(x=60, y=y_pos, width=160)
            cb = Combobox(self, state="readonly", values=opts, font=('Times New Roman', 11))
            cb.current(0)
            cb.place(x=230, y=y_pos, width=240)
            self.combos.append(cb)

    def select_file(self):
        filename = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pkl")])
        if filename:
            self.modelname = filename
            messagebox.showinfo("Model Loaded", f"Using model: {os.path.basename(filename)}")

    def show_prediction(self, result, data):
        suggestions = []
        if data['package'] == 1:
            suggestions.append("Vacuum or Modified Atmosphere Packaging may improve shelf life.")
        if data['sterilization'] == 1:
            suggestions.append("Sterilization methods (low/high temp, microwave) are recommended.")
        if data['temperature'] == 3:
            suggestions.append("Lowering storage temperature is the most effective strategy.")
        if data['preservative'] == 1:
            suggestions.append("Consider adding preservatives to extend shelf life.")
        if not suggestions:
            suggestions.append("Current conditions are optimized.")

        self.result_text.configure(state='normal')
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "Prediction Analysis:\n", "header")
        for item in suggestions:
            self.result_text.insert("end", f"• {item}\n", "bullet")
        self.result_text.configure(state='disabled')

    def show_results_img(self, result):
        if result > 5 or result < 1: result = 0
        try:
            pil_img = Image.open(f'./L{result}.png').resize((350, 90), Image.Resampling.LANCZOS)
            self.res_photo = ImageTk.PhotoImage(pil_img)
            lbl = tk.Label(self, image=self.res_photo, bg="#F0F4C3")
            lbl.place(x=555, y=180, width=350, height=90)
        except: pass

    def getValues(self):
        try:
            vals = [cb.current() + 1 for cb in self.combos]
            data = {'package': vals[0], 'preservative': vals[1], 'temperature': vals[2], 'sterilization': vals[3]}
            result = predict_logic(self.modelname, data)
            if result == -1:
                messagebox.showerror("Error", "Prediction failed. Ensure model and data files exist.")
                return
            self.show_prediction(result, data)
            self.show_results_img(result)
        except Exception as e:
            messagebox.showerror("Error", str(e))


class Retrain(PublicMod):
    def __init__(self):
        super().__init__()
        self.base_window_setup()
        self.initPage()

    def initPage(self):
        self.logo_section(self)
        self.nav_buttons(self, active_idx=3) 

        tk.Canvas(self, bg="#FFF8E1", bd=0).place(x=50, y=160, width=900, height=340)

        # 1. 优化输入框：字体改为 Times New Roman，字号 14
        self.input_text = tk.Text(self, font=FONT_INPUT, wrap="none")
        self.input_text.place(x=70, y=180, width=860, height=220)
        
        # 2. 更新默认 Example 内容
        default_csv = """temperature,package,preservative,sterilization,target
1,3,1,1,3
1,1,1,1,1
2,1,3,1,2
2,1,1,1,1"""
        self.input_text.insert('1.0', default_csv)

        tk.Button(self, text="Upload CSV", font=FONT_BTN, bg=COLOR_UPLOAD, fg="white",
                  command=self.upload_csv).place(x=300, y=430, width=180, height=45)

        tk.Button(self, text="SUBMIT", font=FONT_BTN, bg=COLOR_SUBMIT, fg="white",
                  command=self.perform_retrain).place(x=520, y=430, width=180, height=45)

        self.citation(self)
        self.copyright(self)

    # 3. 自定义错误弹窗 (用于显示大字号 Example)
    def show_custom_error(self, title, message, example_text):
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry("600x450")
        top.transient(self) # 设置为模态窗口
        top.grab_set()

        # 图标和错误信息
        tk.Label(top, text="❌  Data Validation Failed", font=('Times New Roman', 16, 'bold'), fg="red").pack(pady=10)
        tk.Label(top, text=message, font=('Times New Roman', 12), justify="left").pack(pady=5, padx=20)

        # Example 标题
        tk.Label(top, text="Correct Format Example:", font=('Times New Roman', 12, 'bold')).pack(pady=(15, 5))

        # Example 内容区 (文本框)
        txt = tk.Text(top, font=FONT_EXAMPLE, height=6, width=50, bg="#F5F5F5", relief="flat")
        txt.insert("1.0", example_text)
        txt.configure(state="disabled") # 只读
        txt.pack(padx=20)

        # 关闭按钮
        tk.Button(top, text="OK", font=('Times New Roman', 12), command=top.destroy, width=10).pack(pady=20)

        # 居中显示
        self.center_window(top)

    def center_window(self, win):
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f'{width}x{height}+{x}+{y}')

    def upload_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath: return

        try:
            df = pd.read_csv(filepath)
            if not set(REQUIRED_COLUMNS).issubset(df.columns):
                raise ValueError(f"Missing columns. Required: {REQUIRED_COLUMNS}")
            df = df[REQUIRED_COLUMNS]
            for col, valid_vals in VALID_RANGES.items():
                unique_vals = df[col].dropna().unique()
                if not set(unique_vals).issubset(set(valid_vals)):
                    invalid = set(unique_vals) - set(valid_vals)
                    raise ValueError(f"Invalid value in '{col}': {invalid}\nAllowed: {valid_vals}")

            csv_str = df.to_csv(index=False)
            self.input_text.delete('1.0', 'end')
            self.input_text.insert('1.0', csv_str)
            messagebox.showinfo("Valid Data", "CSV file validated and loaded successfully!")

        except Exception as e:
            error_msg = f"Error details: {str(e)}\n\nPlease ensure format matches Table S1."
            example = """temperature,package,preservative,sterilization,target
1,3,1,1,3
1,1,1,1,1
2,1,3,1,2"""
            # 使用自定义弹窗显示大号字体的 Example
            self.show_custom_error("Invalid CSV", error_msg, example)

    def perform_retrain(self):
        raw_data = self.input_text.get('1.0', 'end').strip()
        try:
            new_df = pd.read_csv(io.StringIO(raw_data))
            if not set(REQUIRED_COLUMNS).issubset(new_df.columns):
                raise ValueError("Missing columns.")
            
            if os.path.exists(DATA_FILE_RAW):
                old_df = pd.read_csv(DATA_FILE_RAW)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            combined_df.to_csv(DATA_FILE_RAW, index=False)

            encoder = ce.LeaveOneOutEncoder(cols=['temperature', 'package', 'preservative', 'sterilization'])
            encoded_df = encoder.fit_transform(combined_df.drop('target', axis=1), combined_df['target'])
            encoded_df['target'] = combined_df['target']
            encoded_df.to_csv(DATA_FILE_ENCODED, index=False)

            state, f1, acc = train_logic(MODEL_FILE)

            if state == 1:
                self.show_success_ui(f1, acc)
            else:
                messagebox.showerror("Error", "Training Failed")

        except Exception as e:
            error_msg = f"Processing Error: {str(e)}"
            example = """temperature,package,preservative,sterilization,target
1,3,1,1,3
1,1,1,1,1
2,1,3,1,2"""
            self.show_custom_error("Format Error", error_msg, example)

    def show_success_ui(self, f1, acc):
        messagebox.showinfo("Success", f"Retraining Complete!\nValidation F1-score: {f1:.4f}")
        msg = f"Model Updated. F1: {f1:.4f}  Acc: {acc:.4f}"
        lbl_bg = tk.Label(self, text=msg, font=('Times New Roman', 12, 'bold'), fg="green")
        lbl_bg.place(x=200, y=485, width=600)
        link = tk.Label(self, text="[ Click here to view ROC Curve ]", font=('Times New Roman', 12, 'underline'), 
                        fg="blue", cursor="hand2")
        link.place(x=350, y=510, width=300)
        link.bind("<Button-1>", lambda e: open_file(os.path.join(os.getcwd(), 'ROC.png')))


# === 核心逻辑函数 ===

def train_logic(modelname):
    try:
        if not ensure_encoded_file_exists(): return -1, None, None
        data = pd.read_csv(DATA_FILE_ENCODED)
        X = data.drop('target', axis=1)
        Y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
        grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                            {'n_estimators': [50, 100], 'max_depth': [10, 20]}, 
                            cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        joblib.dump(best_model, modelname)
        y_pred = grid.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        try:
            classes = sorted(list(Y.unique()))
            y_test_bin = label_binarize(y_test, classes=classes)
            y_prob = best_model.predict_proba(X_test)
            plt.figure(figsize=(8, 6))
            for i in range(len(classes)):
                if i < y_test_bin.shape[1] and i < y_prob.shape[1]:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} AUC={auc(fpr, tpr):.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig('ROC.png', dpi=300)
            plt.close()
        except: pass
        return 1, f1, acc
    except Exception as e:
        print(e)
        return -1, None, None

def predict_logic(modelname, newdata):
    try:
        if not ensure_encoded_file_exists(): return -1
        df_raw = pd.read_csv(DATA_FILE_RAW)
        df_enc = pd.read_csv(DATA_FILE_ENCODED)
        input_vec = []
        for col in ['temperature', 'package', 'preservative', 'sterilization']:
            val = newdata[col]
            indices = df_raw.index[df_raw[col] == val].tolist()
            if not indices:
                mean_val = df_enc[col].mean()
            else:
                mean_val = df_enc.iloc[indices][col].mean()
            input_vec.append(mean_val)
        model = joblib.load(modelname)
        return model.predict([input_vec])[0]
    except Exception as e:
        print(e)
        return -1

if __name__ == "__main__":
    app = Main()
    app.mainloop()