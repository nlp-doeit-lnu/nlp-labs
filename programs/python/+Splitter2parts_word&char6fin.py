import os
import re
import unicodedata
import sys
import tkinter as Tk
from tkinter import simpledialog, messagebox
from tkinter.filedialog import askdirectory


class SplitModeDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Split Mode")
        self.label = Tk.Label(master, text="Choose split mode:")
        self.label.pack()

        self.var = Tk.StringVar()
        self.var.set("Characters")

        self.characters_button = Tk.Radiobutton(
            master, text="Characters", variable=self.var, value="Characters"
        )
        self.characters_button.pack(anchor=Tk.W)

        self.words_button = Tk.Radiobutton(
            master, text="Words", variable=self.var, value="Words"
        )
        self.words_button.pack(anchor=Tk.W)

    def apply(self):
        self.result = self.var.get()


def get_control_chars():
    all_chars = (chr(i) for i in range(sys.maxunicode))
    control_chars = ''.join(
        c for c in all_chars if unicodedata.category(c) == 'Cc')
    return control_chars


def remove_control_chars(s):
    control_chars = get_control_chars()
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    return control_char_re.sub('', s)


def split_by_words(text):
    words = text.split()
    half_words = len(words) // 2
    return [' '.join(words[:half_words]), ' '.join(words[half_words:])]


def split_by_chars(text):
    encoded_text = text.encode('utf-8')
    half_length = len(encoded_text) // 2

    # Find the nearest UTF-8 character boundary
    while half_length < len(encoded_text) and (encoded_text[half_length] & 0xC0) == 0x80:
        half_length += 1

    part1 = encoded_text[:half_length].decode('utf-8')
    part2 = encoded_text[half_length:].decode('utf-8')

    return [part1, part2]


def set_results(parts_list, output_directory, filename):
    base_name, ext = os.path.splitext(filename)
    for index, part in enumerate(parts_list, start=1):
        # Add an underscore between index and filename
        new_filename = f'{index}_{base_name}{ext}'
        filepath = os.path.join(output_directory, new_filename)

        original_length = len(part)
        encoded_length = len(part.encode('utf-8'))

        with open(filepath, 'w', encoding='utf-8') as part_to_write:
            part_to_write.write(part)


def main():
    root = Tk.Tk()
    root.withdraw()

    input_directory = askdirectory(initialdir="/", title="Select directory")

    # Create a custom dialog for split mode
    split_mode_dialog = SplitModeDialog(root)
    split_mode = split_mode_dialog.result

    if not split_mode:
        messagebox.showerror("Error", "Split mode not selected.")
        return

    # Ask whether to choose the output folder
    output_choice = messagebox.askyesno(
        "Output Folder", "Do you want to choose the output folder?")

    if output_choice:
        output_directory = askdirectory(
            initialdir="/", title="Select output directory")
    else:
        output_directory = input_directory

    os.makedirs(output_directory, exist_ok=True)

    for entry in os.scandir(input_directory):
        filename = entry.name
        if filename.lower().endswith((".txt", ".TXT")) and entry.is_file():
            input_file = entry.path

            lowercase_filename = filename.lower()

            try:
                with open(input_file, "r", encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError as e:
                print(f"Error reading file {filename} ({e}). Skipping.")
                messagebox.showwarning(
                    "Warning", f"Cannot read {filename}. Skipping.")
                continue

            print(f"Original length of {filename}: {len(text)}")

            text = text.strip()
            text = remove_control_chars(text)

            if split_mode.lower() == "characters":
                parts = split_by_chars(text)
            elif split_mode.lower() == "words":
                parts = split_by_words(text)

            for index, part in enumerate(parts, start=1):
                print(f"Length of part {index}: {len(part)}")

            set_results(parts, output_directory, lowercase_filename)

            for index, part in enumerate(parts, start=1):
                new_filename = f'{index}_{lowercase_filename}'
                filepath = os.path.join(output_directory, new_filename)
                size = os.path.getsize(filepath)
                print(f"Size of {new_filename}: {size} bytes")

    messagebox.showinfo("Information", "Done!")


if __name__ == '__main__':
    main()
