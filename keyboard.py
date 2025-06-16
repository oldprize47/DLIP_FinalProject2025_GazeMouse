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


def backspace():
    current = entry.get()
    if current:
        entry.delete(len(current) - 1, tk.END)
        pyautogui.press("backspace")


# 모니터 크기 측정
screen_width, screen_height = pyautogui.size()
window_width = screen_width
window_height = screen_height // 2

root = tk.Tk()
root.title("Simple Virtual Keyboard")
# 아래쪽 절반에 배치
root.geometry(f"{window_width}x{window_height}+0+{screen_height//2 - 100}")
root.resizable(False, False)

# ── grid 가중치(열/행 모두) 설정: 꽉 차게!
for i in range(12):  # 최대 열 개수만큼(버튼 개수만큼)
    root.grid_columnconfigure(i, weight=1)
for i in range(6):  # 충분한 행
    root.grid_rowconfigure(i, weight=1)

# ── 입력창 (가로 전체, 위쪽 행) ──
entry = tk.Entry(root, font=("Arial", 32), bg="white", justify="left")
entry.grid(row=0, column=0, columnspan=12, padx=8, pady=(12, 8), sticky="nsew")

# ── 알파벳 버튼(3줄로 예시) ──
letters = "QWERTYUIOP!ASDFGHJKLZXCVBNM"
buttons = []
btn_font = ("Arial", 32)

for idx, char in enumerate(letters):
    if idx < 11:  # 첫 줄
        row, col = 1, idx
    elif 11 <= idx <= 21:  # 둘째 줄
        row, col = 2, idx - 11
    else:  # 셋째 줄
        row, col = 3, idx - 22
    btn = tk.Button(root, text=char, font=btn_font, command=lambda c=char: type_letter(c))
    btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
    buttons.append(btn)

# ── Space (아래 넓게) ──
tk.Button(root, text="Space", font=btn_font, command=type_space).grid(row=4, column=2, columnspan=6, padx=4, pady=8, sticky="nsew")

# ── Clear (오른쪽) ──
tk.Button(root, text="Clear", font=btn_font, command=clear_entry).grid(row=3, column=9, padx=4, pady=4, sticky="nsew")

# ── Backspace (오른쪽) ──
tk.Button(root, text="⌫", font=btn_font, command=backspace).grid(row=3, column=10, padx=4, pady=4, sticky="nsew")

root.mainloop()
