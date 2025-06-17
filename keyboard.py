import tkinter as tk
import pyautogui


def type_letter(letter):
    """
    Insert a letter into the entry box.
    """
    entry.insert(tk.END, letter)


def type_space():
    """
    Insert a space character.
    """
    entry.insert(tk.END, " ")


def clear_entry():
    """
    Clear the entry box.
    """
    entry.delete(0, tk.END)


def backspace():
    """
    Remove the last character.
    """
    current = entry.get()
    if current:
        entry.delete(len(current) - 1, tk.END)


# Get monitor size
screen_width, screen_height = pyautogui.size()
window_width = screen_width
window_height = screen_height // 2

root = tk.Tk()
root.title("Simple Virtual Keyboard")
# Position at the lower half of the screen
root.geometry(f"{window_width}x{window_height}+0+{screen_height//2 - 100}")
root.resizable(False, False)

# ESC key exits program
root.bind("<Escape>", lambda event: root.destroy())

# Set grid weights for even distribution (columns/rows)
for i in range(12):  # Up to 12 columns (for buttons)
    root.grid_columnconfigure(i, weight=1)
for i in range(6):  # Enough rows
    root.grid_rowconfigure(i, weight=1)

# Entry box (top row, spans all columns)
entry = tk.Entry(root, font=("Arial", 32), bg="white", justify="left")
entry.grid(row=0, column=0, columnspan=12, padx=8, pady=(12, 8), sticky="nsew")

# Alphabet buttons (3 rows as example)
letters = "QWERTYUIOP!ASDFGHJKLZXCVBNM"
buttons = []
btn_font = ("Arial", 32)

for idx, char in enumerate(letters):
    # First row: 0~10, second row: 11~21, third row: 22+
    if idx < 11:
        row, col = 1, idx
    elif 11 <= idx <= 21:
        row, col = 2, idx - 11
    else:
        row, col = 3, idx - 22
    btn = tk.Button(root, text=char, font=btn_font, command=lambda c=char: type_letter(c))
    btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
    buttons.append(btn)

# Space button (wide at the bottom)
tk.Button(root, text="Space", font=btn_font, command=type_space).grid(row=4, column=2, columnspan=6, padx=4, pady=8, sticky="nsew")

# Clear button (right side)
tk.Button(root, text="Clear", font=btn_font, command=clear_entry).grid(row=3, column=9, padx=4, pady=4, sticky="nsew")

# Backspace button (right side)
tk.Button(root, text="⌫", font=btn_font, command=backspace).grid(row=3, column=10, padx=4, pady=4, sticky="nsew")

root.mainloop()
