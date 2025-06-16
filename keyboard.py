import tkinter as tk
import pyautogui

def type_letter(letter):
    entry.insert(tk.END, letter)
    pyautogui.write(letter)

def type_space():
    entry.insert(tk.END, " ")
    pyautogui.write(" ")

def clear_entry():
    entry.delete(0, tk.END)

def backspace():                     # ← Backspace 기능
    current = entry.get()
    if current:
        entry.delete(len(current) - 1, tk.END)
        pyautogui.press("backspace")  # 실제 키보드 입력도 삭제

root = tk.Tk()
root.title("Simple Virtual Keyboard")

# ── 입력창 ──
entry = tk.Entry(root, width=30, font=("Arial", 18), bg="white")
entry.grid(row=0, column=0, columnspan=12, padx=5, pady=(12, 5))

# ── 알파벳 버튼 ──
letters = "QWERTYUIOP!ASDFGHJKLZXCVBNM"
for idx, char in enumerate(letters):
    if idx < 10:
        row, col = 1, idx
    elif idx == 10:
        row, col = 1, 10
    elif 11 <= idx <= 20:
        row, col = 2, idx - 11
    else:
        row, col = 3, idx - 21
    tk.Button(root, text=char, width=4, height=2,
              font=("Arial", 14),
              command=lambda c=char: type_letter(c)
              ).grid(row=row, column=col, padx=2, pady=2)

# ── Space ──
tk.Button(root, text="Space", width=18, height=2,
          font=("Arial", 14), command=type_space
          ).grid(row=4, column=2, columnspan=6, padx=2, pady=8)

# ── Clear ──
tk.Button(root, text="Clear", width=4, height=2,
          font=("Arial", 14), command=clear_entry
          ).grid(row=3, column=7, padx=2, pady=2)

# ── Backspace ──  (row 3, col 8 위치 예시)
tk.Button(root, text="⌫", width=4, height=2,
          font=("Arial", 14), command=backspace
          ).grid(row=3, column=8, padx=2, pady=2)

root.geometry("+400+400")
root.mainloop()
