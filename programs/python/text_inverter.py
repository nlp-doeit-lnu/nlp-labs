from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox as mb

import regex as re
import time
import itertools
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
import numpy as np

import multiprocessing
from functools import partial

SIZE_OF_TEXT_TO_SHOW = 1000
toktok = ToktokTokenizer()

# File operations functions
def open_file():
    opened_files = filedialog.askopenfilenames(filetypes=[("txt-file", "*.txt")])
    for file in opened_files:
        cleared_file = re.findall(r'[^\/]+(?=\.)', file)
        files_listbox.insert(END, cleared_file)
        opened_files_list.append(file)
    

def close_file():
    if len(files_listbox.curselection()) == 0:
        mb.showerror("OK", "Виберіть спочатку текст зі списку")
    else:
        selected_file = files_listbox.curselection()[0]
        opened_files_list.pop(selected_file)
        files_listbox.delete(selected_file)
        preview_text_area.delete("1.0", "end")
        processed_text_area.delete("1.0", "end")


def close_all_files():
    opened_files_list.clear()
    files_listbox.delete(0, END)
    preview_text_area.delete("1.0", "end")
    processed_text_area.delete("1.0", "end")


def save_file():
    if len(files_listbox.curselection()) == 0:
        mb.showerror("OK", "Виберіть спочатку текст зі списку")
    elif len(processed_text.get()) == 0:
        mb.showerror("OK", "Спочатку інтвертуйте текст")
    else:
        selected_file = files_listbox.curselection()[0]
        text_name = files_listbox.get(selected_file)
        save_dir = filedialog.askdirectory(title="Виберіть директорію для збереження файлу")
        new_filename = save_dir + "/" + text_name[0] + "_I" + invertion_type.get()[0].upper() + ".txt"
        with open(new_filename, "w", encoding="utf-8") as f:
            f.write(processed_text.get())
        mb.showinfo("OK", f"Результати збережені за шляхом {new_filename}")


# Text operations functions
def invert_all_and_save():
    if len(opened_files_list) == 0:
        mb.showerror("OK", "Відкрийте спочатку тексти для інтвертування")
    else:
        text = StringVar()
        finished_text = StringVar()
        increment_step = 100 / len(opened_files_list)
        save_dir = filedialog.askdirectory(title="Виберіть директорію для збереження файлу")
        start_clock = time.time()
        for txt in opened_files_list:
            with open(txt,"r", encoding="utf-8") as f:
                text.set(f.read())

            if invertion_type.get() == "character":
                text_to_process = re.sub('\s{2,}', " ", text.get())

                inverted_text = text_to_process[::-1]
                finished_text.set(inverted_text)            
            elif invertion_type.get() == "word":
                text_to_process = re.sub('\s{2,}', " ", text.get())
                preprocessed_text = re.sub(r"[…;.,?!‼¡¿§()""''@$%:]", "", text_to_process)
                
                chunk_size = round(len(preprocessed_text)/multiprocessing.cpu_count())
                chunks = [x for x in range(0, len(preprocessed_text), chunk_size)]
                pool = multiprocessing.Pool(initializer=init_worker, initargs=(chunk_size,))
                func = partial(run_process_word, preprocessed_text)
                tokenized_text = pool.map(func, chunks)
                pool.close()
                pool.join()
                
                tokenized_text = list(itertools.chain.from_iterable(tokenized_text))
                finished_text.set(" ".join(tokenized_text[::-1]))
            elif invertion_type.get() == "sentence":
                text_to_process = re.sub('\s{2,}', " ", text.get())

                generated_regex = generate_regex_with_user_delims()
                preprocessed_text = re.split(rf"{generated_regex}", text_to_process)
                finished_text.set(" ".join(preprocessed_text[::-1]))
            
            txt = re.findall(r'[^\/]+(?=\.)', txt)
            new_filename = save_dir + "/" + txt[0] + "_I" + invertion_type.get()[0].upper() + ".txt"
            with open(new_filename, "w", encoding="utf-8") as f:
                f.write(finished_text.get())
            progress_bar["value"] += increment_step
            root.update_idletasks()
        end_clock = time.time()
        execution_time_label.config(text=f"Час інверсії: {round(end_clock - start_clock, 3)}") 
        mb.showinfo("OK", f"Усі тексти інтвертовано та збережено у папку {save_dir}!\n Час роботи програми: {round(end_clock - start_clock, 3)} с.")
        progress_bar["value"] = 0


def show_file_content(event):
    selected_file = opened_files_list[files_listbox.curselection()[0]]
    with open(selected_file, "r", encoding="utf-8") as f:
        file_content = f.read()
    original_text.set(file_content)
    processed_text_area.delete("1.0", "end")
    preview_text_area.delete("1.0", "end")
    if bool(show_full_text.get()):
        preview_text_area.insert("1.0", file_content)
    else:
        preview_text_area.insert("1.0", file_content[:SIZE_OF_TEXT_TO_SHOW])
    

def init_worker(data):
    global CHUNK_SIZE
    CHUNK_SIZE = data


# def run_process_char(splitted_text, start):
#     data = splitted_text[start:start+CHUNK_SIZE]
#     tokenized_text = [[toktok.tokenize(w), " "] for w in data]
#     finished_text = list(itertools.chain(*list(itertools.chain(*tokenized_text))))
#     return finished_text


def run_process_word(text, start):
    data = text[start:start+CHUNK_SIZE]
    tokenized_text = toktok.tokenize(data)
    return tokenized_text


def generate_wrappers(text):
    text = re.sub("\n", " ", text)
    text = re.sub('\s{2,}', " ", text)
    n_chars = 40
    result = ""
    index = 0
    while index < len(text):
        result += text[index:index+n_chars] + "\n"
        index += n_chars
    return result


def generate_regex_with_user_delims():
    users_delimeter = list(set(delimeter_entry.get().split(" ")))
    users_delimeter = [delim for delim in users_delimeter if len(delim) != 0]
    if len(users_delimeter) < 1:
        regex = """(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\…|\."|\?"|\!"|\…"|\.'|\?'|\!'|\…'|\.>|\?>|\!>|\…>|\.»|\?»|\!»|\…»|\.”|\?”|\!”|\…” |\.„|\?„|\!„|\…„)\s+"""
    else:
        regex = """(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\…|\."|\?"|\!"|\…"|\.'|\?'|\!'|\…'|\.>|\?>|\!>|\…>|\.»|\?»|\!»|\…»|\.”|\?”|\!”|\…” |\.„|\?„|\!„|\…„"""
        for delim in users_delimeter:
            regex += "|\\" + delim
        regex += ')\s+'
    return regex


def invert_text():
    if len(files_listbox.curselection()) == 0:
        mb.showerror("OK", "Виберіть спочатку текст зі списку")
    else:
        text_to_process = original_text.get()
        if invertion_type.get() == "character":
            start_process = time.time()

            text_to_process = re.sub('\s{2,}', " ", text_to_process)

            inverted_text = text_to_process[::-1]
            processed_text.set(inverted_text)
            
            inverted_text = generate_wrappers(inverted_text)
            inverted_wrapped_text.set(inverted_text)

            end_process = time.time()
            execution_time_label.config(text=f"Час інверсії: {round(end_process - start_process, 3)}")
        elif invertion_type.get() == "word":
            start_process = time.time()

            text_to_process = re.sub('\s{2,}', " ", text_to_process)
            preprocessed_text = re.sub(r"[…;.,?!‼¡¿§()""''@$%:]", "", text_to_process)

            chunk_size = round(len(preprocessed_text)/multiprocessing.cpu_count())
            chunks = [x for x in range(0, len(preprocessed_text), chunk_size)]
            pool = multiprocessing.Pool(initializer=init_worker, initargs=(chunk_size,))
            func = partial(run_process_word, preprocessed_text)
            tokenized_text = pool.map(func, chunks)
            pool.close()
            pool.join()
            tokenized_text = list(itertools.chain.from_iterable(tokenized_text))

            inverted_text = " ".join(tokenized_text[::-1])
            processed_text.set(inverted_text)

            inverted_text = generate_wrappers(inverted_text)
            inverted_wrapped_text.set(inverted_text)

            end_process = time.time()
            execution_time_label.config(text=f"Час інверсії: {round(end_process - start_process, 3)}")
        elif invertion_type.get() == "sentence":
            start_process = time.time()

            text_to_process = re.sub('\s{2,}', " ", text_to_process)
            generated_regex = generate_regex_with_user_delims()
            preprocessed_text = re.split(rf"{generated_regex}", text_to_process)
           
            inverted_text = " ".join(preprocessed_text[::-1])
            processed_text.set(inverted_text)

            inverted_text = generate_wrappers(inverted_text)
            inverted_wrapped_text.set(inverted_text)

            end_process = time.time()
            execution_time_label.config(text=f"Час інверсії: {round(end_process - start_process, 3)}")
        processed_text_area.delete("1.0", "end")
        if bool(show_full_text.get()):
            processed_text_area.insert("1.0", inverted_wrapped_text.get())
        else:
            processed_text_area.insert("1.0", inverted_wrapped_text.get()[:SIZE_OF_TEXT_TO_SHOW])


def update_text_areas():
    processed_text_area.delete("1.0", "end")
    preview_text_area.delete("1.0", "end")
    file_content = original_text.get()
    processed_content = processed_text.get()
    if bool(show_full_text.get()):
        preview_text_area.insert("1.0", file_content)
        processed_text_area.insert("1.0", inverted_wrapped_text.get())
    else:
        processed_text_area.insert("1.0", inverted_wrapped_text.get()[:SIZE_OF_TEXT_TO_SHOW])
        preview_text_area.insert("1.0", file_content[:SIZE_OF_TEXT_TO_SHOW])
        


def show_delimeter_entry():
    if invertion_type.get() == "sentence":
        delimeter_entry.configure(state="normal")
    else:
        delimeter_entry.configure(state="disabled")


if __name__ == "__main__":
    root = Tk()
    root.title("Text Inverter 2.0")
    root.geometry('1080x520')

    # Show folder content, manage selected texts, apply text from selected file to text area
    files_listbox_frame = Frame(root, padx=10, pady=10)
    files_listbox_frame.place(relx=0.01, rely=0.01, relheight=0.98, relwidth=0.15)

    opened_files_list = []
    files_listbox = Listbox(files_listbox_frame)
    files_listbox.place(relx=0.01, rely=0.01, relheight=0.98, relwidth=0.88)
    files_listbox.bind("<<ListboxSelect>>", show_file_content)
    scrollbar = Scrollbar(files_listbox_frame, command=files_listbox.yview)
    scrollbar.place(relx=0.88, rely=0.01, relheight=0.98, relwidth=0.1)
    files_listbox.config(yscrollcommand=scrollbar.set)

    # Storing original and processed text
    original_text = StringVar()
    processed_text = StringVar()
    inverted_wrapped_text = StringVar()

    # Text area to preview original text
    preview_text_frame = Frame(root, padx=10, pady=10)
    preview_text_frame.place(relx=0.2, rely=0.01, relheight=0.98, relwidth=0.35)

    preview_text_area = scrolledtext.ScrolledText(preview_text_frame)
    preview_text_area.place(relx=0.01, rely=0.01, relheight=0.45, relwidth=0.98)

    # Text area to preview processed text
    processed_text_frame = Frame(root, padx=10, pady=10)
    processed_text_frame.place(relx=0.6, rely=0.01, relheight=0.98, relwidth=0.35)

    processed_text_area = scrolledtext.ScrolledText(processed_text_frame)
    processed_text_area.place(relx=0.01, rely=0.01, relheight=0.45, relwidth=0.98)

    # Invertor type
    invertion_type_label = ttk.Label(preview_text_frame, text="Режим інверсії:")
    invertion_type_label.place(relx=0.01, rely=0.48, relheight=0.04, relwidth=0.25)

    invertion_type = StringVar(value="character")
    character_button = ttk.Radiobutton(preview_text_frame, text="Символи", value="character", variable=invertion_type, command=show_delimeter_entry)
    character_button.place(relx=0.01, rely=0.53, relheight=0.04)

    word_button = ttk.Radiobutton(preview_text_frame, text="Слова", value="word", variable=invertion_type, command=show_delimeter_entry)
    word_button.place(relx=0.01, rely=0.57, relheight=0.04)

    sentence_button = ttk.Radiobutton(preview_text_frame, text="Речення", value="sentence", variable=invertion_type, command=show_delimeter_entry)
    sentence_button.place(relx=0.01, rely=0.61, relheight=0.04)

    # Limiting size of the text showed
    show_full_text = IntVar(value=0)
    text_limitting_button = ttk.Checkbutton(preview_text_frame, text="Відображати\nвесь текст", variable=show_full_text,
                                            onvalue=1, offvalue=0, command=update_text_areas)
    text_limitting_button.place(relx=0.01, rely=0.66)

    # Specify delimeters from users
    delimeter_label = Label(preview_text_frame, text="Знаки-роздільники речень\n (через пробіл, '.', '!', '?', '…' вже включені)")
    delimeter_label.place(relx=0.3, rely=0.75)

    delimeter_entry = Entry(preview_text_frame, state="disabled")
    delimeter_entry.place(relx=0.3, rely=0.85)


    # Operation buttons
    open_button = ttk.Button(preview_text_frame, text="Відкрити файл(и)", command=open_file)
    open_button.place(relx=0.3, rely=0.48, width=110, height=25)

    invert_text_button = ttk.Button(preview_text_frame, text="Інвертувати\nвибраний\nтекст", command=invert_text)
    invert_text_button.place(relx=0.3, rely=0.54, width=110, height=55)

    save_button = ttk.Button(preview_text_frame, text="Зберегти\nрезультат", command=save_file)
    save_button.place(relx=0.3, rely=0.66, width=110, height=40)


    close_button = ttk.Button(preview_text_frame, text="Закрити файл(и)", command=close_file)
    close_button.place(relx=0.65, rely=0.48, width=110, height=25)

    invert_all_and_save_button = ttk.Button(preview_text_frame, text="Інвертувати всі\nтексти та зберегти\nрезультати", command=invert_all_and_save)
    invert_all_and_save_button.place(relx=0.65, rely=0.54, width=110, height=55)

    close_all_button = ttk.Button(preview_text_frame, text="Закрити\nусі файли", command=close_all_files)
    close_all_button.place(relx=0.65, rely=0.66, width=110, height=40)

    # Progress bar
    progress_bar = ttk.Progressbar(processed_text_frame, orient=HORIZONTAL, length=100, mode="determinate")
    progress_bar.place(relx=0.01, rely=0.48, height=20, relwidth=0.98)

    # Invertion time
    execution_time_label = ttk.Label(processed_text_frame, text="Час інверсії:")
    execution_time_label.place(relx=0.01, rely=0.52)


    root.mainloop()