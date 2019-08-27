# -*- coding: utf-8 -*-
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


class GUI(object):
    def __init__(self, root):
        self.master = root
        self.master.title("Huawei GUI")
        self.master.geometry("300x300")
        self.master.configure(background='white')
        self.master.resizable(width=False, height=False)

        self.btn = ttk.Button(root, text="选择进行超分的图片", command=self.xz)
        self.btn.place(x=50, y=20)
        self.lb = tk.Label(root, text = '', width=23)
        self.lb.place(x=20, y=50)
        self.info_label = tk.Label(self.master, text='目前支持格式：\njpg, png, bmp.\n一次最多选择5张图片', bg='white', font=('Arial', 12), width=20).place(x=20,y=220)

        self.func = ttk.Button(self.master, text='一键超分', width=10, command=None)
        self.func.place(x = 200, y = 220)

    def xz(self):
        filenames = tk.filedialog.askopenfilenames()
        if len(filenames) > 5:
            filenames = filenames[:5]
        if len(filenames) != 0:
            string_filename = ""
            output_name = ""
            for i in range(0, len(filenames)):
                if not filenames[i].endswith(".png") or filenames[i].endswith(".jpg") or filenames[i].endswith(".bmp"):
                    continue
                string_filename += str(filenames[i]) + "\n"
                output_name += filenames[i].split('/')[-1] + "\n"
            self.lb.config(text = "您选择的文件是\n" + output_name)
        else:
            self.lb.config(text = "您没有选择任何文件")


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    gui.master.mainloop()
