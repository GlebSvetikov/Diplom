import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from config import AppConfig
from ui.base_view import BaseView

class MainView(BaseView):
    def __init__(self, root: tk.Tk, controller):
        self.root = root
        self.controller = controller
        self.config = AppConfig()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Оценка профпригодности курсантов")
        self.root.configure(bg=self.config.palette["bg"])
        self.root.geometry("1100x600")

        self._setup_title()
        self._setup_buttons()
        self._setup_context_menu()

    def _setup_title(self):
        title = tk.Label(
            self.root, 
            text="Система оценки профпригодности курсантов",
            font=self.config.fonts["title"],
            bg=self.config.palette["bg"],
            fg=self.config.palette["fg"]
        )
        title.pack(pady=15)

    def _setup_buttons(self):
        btn_frame = tk.Frame(self.root, bg=self.config.palette["bg"])
        btn_frame.pack(pady=20)

        for text, command in self.config.buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=getattr(self.controller, command),
                bg=self.config.palette["button_bg"],
                fg=self.config.palette["button_fg"],
                font=self.config.fonts["button"],
                relief='raised'
            )
            btn.pack(fill='x', pady=8, ipady=8)

    def _setup_context_menu(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(
            label="Оценить", 
            command=self.controller.evaluate_cadet
        )
        self.context_menu.add_command(
            label="Удалить", 
            command=self.controller.delete_cadet
        )
        self.context_menu.add_command(
            label="Редактировать", 
            command=self.controller.edit_cadet
        )

    def show_learning_curve(self, train_acc: list, test_acc: list):
        window = tk.Toplevel(self.root)
        window.title("Кривая обучения")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_acc, label='Обучающая выборка', linewidth=2)
        ax.plot(test_acc, label='Тестовая выборка', linewidth=2)

        ax.set_title("График обучения", fontsize=16)
        ax.set_xlabel("Эпохи", fontsize=14)
        ax.set_ylabel("Точность", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill='both', expand=True)

    def show_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        window = tk.Toplevel(self.root)
        window.title("Матрица ошибок")

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap='Blues')

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha='center', va='center', fontsize=14, weight='bold')

        ax.set_xlabel('Предсказано', fontsize=14)
        ax.set_ylabel('Истинные метки', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Не отчислен', 'Отчислен'], fontsize=12)
        ax.set_yticklabels(['Не отчислен', 'Отчислен'], fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        fig.colorbar(cax)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
