import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import time
from config import AppConfig
from data_model import DataModel
from ml_model import MLModel
from ui.main_view import MainView
from ui.forms import AddCadetForm, EditCadetForm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

class AppController:
    def __init__(self, root: tk.Tk):
        self.view = MainView(root, self)
        self.data_model = DataModel()
        self.ml_model = MLModel()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.data_model.load_training_data(file_path)
                messagebox.showinfo("Успех", "Данные успешно загружены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def train_model(self):
        if self.data_model.training_data is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите данные")
            return
        try:
            X_train, X_test, y_train, y_test = self.data_model.preprocess_data()
            self.ml_model.X_test = X_test
            self.ml_model.y_test = y_test

            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

            self.ml_model.model = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1, warm_start=True, random_state=42)
            self.ml_model.train_accuracies.clear()
            self.ml_model.test_accuracies.clear()

            log_lines = []
            best_test_acc = 0
            epochs_without_improvement = 0
            early_stopped = False
            start_time = time.time()

            for epoch in range(1, 101):
                self.ml_model.model.fit(X_train, y_train)
                y_pred_train = self.ml_model.model.predict(X_train)
                y_pred_test = self.ml_model.model.predict(X_test)

                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)

                self.ml_model.train_accuracies.append(train_acc)
                self.ml_model.test_accuracies.append(test_acc)

                log_lines.append(f"Эпоха {epoch:03d}: обуч.={train_acc:.4f}, тест={test_acc:.4f}")

                if test_acc > best_test_acc + 0.001:
                    best_test_acc = test_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= 10:
                    early_stopped = True
                    log_lines.append(f"\u26d4 Остановка на {epoch}-й эпохе: нет улучшений за 10 эпох.")
                    break

            total_time = time.time() - start_time

            cv_scores = cross_val_score(
                MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=200, random_state=42),
                X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            log_win = tk.Toplevel(self.view.root)
            log_win.title("Лог обучения")
            log_win.geometry("640x420")

            txt = tk.Text(log_win, font=("Courier New", 10), bg="#f0f0f0")
            txt.pack(expand=True, fill='both', padx=10, pady=10)

            for line in log_lines:
                txt.insert("end", line + "\n")

            txt.insert("end", f"\n⏱ Время обучения: {total_time:.2f} сек.")
            if early_stopped:
                txt.insert("end", "\n✅ Обучение остановлено досрочно.\n")
            txt.insert("end", f"\nКросс-валидация (5 фолдов):\nСредняя точность: {cv_mean:.4f}, отклонение: {cv_std:.4f}")
            txt.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")

    def show_learning_curve(self):
        if not self.ml_model.train_accuracies:
            messagebox.showwarning("Ошибка", "Сначала обучите модель")
            return
        self.view.show_learning_curve(self.ml_model.train_accuracies, self.ml_model.test_accuracies)

    def show_confusion_matrix(self):
        if self.ml_model.model is None:
            messagebox.showwarning("Ошибка", "Сначала обучите модель")
            return
        y_pred = self.ml_model.model.predict(self.ml_model.X_test)
        self.view.show_confusion_matrix(self.ml_model.y_test, y_pred)

    def add_cadet_manually(self):
        def submit_callback(data):
            self.data_model.add_cadet(data)
            messagebox.showinfo("Успех", "Курсант успешно добавлен")

        AddCadetForm(self.view.root, self.data_model.feature_order,
                     {k: le.classes_.tolist() for k, le in self.data_model.label_encoders.items()},
                     submit_callback).grab_set()

    def add_cadets_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        try:
            new_data = pd.read_excel(file_path)
            new_data['Уровень успешности в ЛП2'] = 'не оценен'
            self.data_model.cadets_data = pd.concat([self.data_model.cadets_data, new_data], ignore_index=True)
            self.data_model._reorder_columns()
            messagebox.showinfo("Успех", f"Добавлено {len(new_data)} курсантов")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def show_cadets_table(self):
        if self.data_model.cadets_data.empty:
            messagebox.showinfo("Информация", "Нет добавленных курсантов")
            return

        window = tk.Toplevel(self.view.root)
        window.title("Список курсантов")
        window.geometry("900x450")

        container = ttk.Frame(window)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        h_scroll = ttk.Scrollbar(container, orient='horizontal')
        v_scroll = ttk.Scrollbar(container, orient='vertical')
        h_scroll.pack(side='bottom', fill='x')
        v_scroll.pack(side='right', fill='y')

        cols = list(self.data_model.cadets_data.columns)
        self.tree = ttk.Treeview(
            container, columns=cols, show='headings',
            xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set,
            selectmode='extended')
        h_scroll.config(command=self.tree.xview)
        v_scroll.config(command=self.tree.yview)

        self.tree.tag_configure('not_evaluated', background='#eeeeee')
        self.tree.tag_configure('eligible', background='#d4edda')
        self.tree.tag_configure('ineligible', background='#f8d7da')

        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center', stretch=True)

        for idx, row in self.data_model.cadets_data.iterrows():
            values = list(row)
            status = row.get('Уровень успешности в ЛП2', 'не оценен')
            tag = ('eligible' if status == "не отчислен" else 'ineligible' if status == "отчислен" else 'not_evaluated')
            self.tree.insert('', 'end', values=values, tags=(tag,))

        self.tree.pack(fill='both', expand=True)
        self.tree.bind("<Button-3>", self._show_context_menu)

        def _clear_selection(event):
            item = self.tree.identify_row(event.y)
            if not item:
                self.tree.selection_remove(self.tree.selection())

        self.tree.bind("<Button-1>", _clear_selection, add="+")

    def _show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            if item not in self.tree.selection():
                self.tree.selection_set(item)
            self.view.context_menu.post(event.x_root, event.y_root)

    def evaluate_cadet(self):
        if self.ml_model.model is None:
            messagebox.showwarning("Ошибка", "Сначала обучите модель")
            return
        if not hasattr(self, 'tree') or not self.tree.selection():
            return

        for item in self.tree.selection():
            idx = self.tree.index(item)
            cadet_data = self.data_model.cadets_data.iloc[idx].to_dict()

            try:
                input_data = []
                for col in self.data_model.feature_order:
                    val = cadet_data[col]
                    if col in self.data_model.label_encoders:
                        val = self.data_model.label_encoders[col].transform([val])[0]
                    else:
                        val = float(val)
                    input_data.append(val)

                pred = self.ml_model.predict(input_data, self.data_model.scaler)
                result = "не отчислен" if pred == 1 else "отчислен"
                self.data_model.cadets_data.at[idx, 'Уровень успешности в ЛП2'] = result
                self.data_model._reorder_columns()
                self.tree.set(item, 'Уровень успешности в ЛП2', result)

                tag = 'eligible' if result == "не отчислен" else 'ineligible'
                stripe = 'evenrow' if idx % 2 == 0 else 'oddrow'
                self.tree.item(item, tags=(stripe, tag))

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при оценке курсанта #{idx+1}: {e}")

    def delete_cadet(self):
        if not hasattr(self, 'tree') or not self.tree.selection():
            messagebox.showinfo("Удаление", "Выберите одного или нескольких курсантов для удаления")
            return

        selected = self.tree.selection()
        confirm = messagebox.askyesno("Подтверждение", f"Удалить {len(selected)} курсантов?")
        if not confirm:
            return

        indices = sorted([self.tree.index(item) for item in selected], reverse=True)
        for idx in indices:
            self.data_model.cadets_data = self.data_model.cadets_data.drop(idx)

        self.data_model.cadets_data.reset_index(drop=True, inplace=True)
        for item in selected:
            self.tree.delete(item)

        messagebox.showinfo("Удаление", f"Удалено {len(selected)} курсантов")

    def edit_cadet(self):
        if not hasattr(self, 'tree') or not self.tree.selection():
            return

        selected = self.tree.selection()
        if len(selected) > 1:
            messagebox.showwarning("Ошибка", "Можно редактировать только одного курсанта за раз.")
            return

        item = selected[0]
        idx = self.tree.index(item)
        cadet = self.data_model.cadets_data.iloc[idx].to_dict()

        def on_submit(updated_data):
            self.data_model.cadets_data.iloc[idx] = updated_data
            for col in self.data_model.cadets_data.columns:
                self.tree.set(item, col, updated_data.get(col, ""))
            tag = 'eligible' if updated_data["Уровень успешности в ЛП2"] == "не отчислен" else 'ineligible' if updated_data["Уровень успешности в ЛП2"] == "отчислен" else 'not_evaluated'
            stripe = 'evenrow' if idx % 2 == 0 else 'oddrow'
            self.tree.item(item, tags=(stripe, tag))

        EditCadetForm(
            self.view.root,
            self.data_model.feature_order,
            {k: le.classes_.tolist() for k, le in self.data_model.label_encoders.items()},
            cadet,
            on_submit
        ).grab_set()

    def export_to_excel(self):
        if self.data_model.cadets_data.empty:
            messagebox.showinfo("Информация", "Нет данных для экспорта")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel файлы", "*.xlsx")],
            title="Сохранить как"
        )
        if not file_path:
            return

        try:
            export_df = self.data_model.cadets_data[
                self.data_model.cadets_data['Уровень успешности в ЛП2'].isin(['отчислен', 'не отчислен'])
            ].copy()
            self.data_model._reorder_columns()
            export_df.to_excel(file_path, index=False)
            messagebox.showinfo("Успех", f"Экспорт завершён: {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
