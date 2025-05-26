import tkinter as tk
from tkinter import ttk, messagebox
from config import AppConfig


class StudentCardForm(tk.Toplevel):
    def __init__(self, parent, feature_order, label_categories, callback, cadet_data=None):
        super().__init__(parent)
        self.title("Карточка курсанта")
        self.configure(bg=AppConfig.palette["bg"])
        self.feature_order = feature_order
        self.label_categories = label_categories
        self.callback = callback
        self.entries = {}
        self.cadet_data = cadet_data or {}
        self._setup_ui()

    def _setup_ui(self):
        self.geometry("1000x700")
        style = ttk.Style(self)
        style.configure("Bold.TLabelframe.Label", font=("Verdana", 12, "bold"))
        style.configure("Bold.TLabelframe", background="#eaf2f8", borderwidth=2, relief="solid")
        self.update_idletasks()
        x = (self.winfo_screenwidth() - self.winfo_width()) // 2
        y = (self.winfo_screenheight() - self.winfo_height()) // 3
        self.geometry(f"+{x}+{y}")

        outer_frame = ttk.Frame(self, padding=20)
        outer_frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(outer_frame, bg=AppConfig.palette["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        grouped_fields = self._group_fields()

        validate_cmd = self.register(self._validate_and_color_field)

        row = 0
        for group_name, fields in grouped_fields.items():
            lf = ttk.LabelFrame(inner_frame, text=group_name, padding=(12, 8), style="Bold.TLabelframe")
            lf.grid(row=row, column=0, padx=16, pady=12, sticky="ew")
            lf.columnconfigure(1, weight=1)

            for i, col in enumerate(fields):
                if col not in self.feature_order:
                    continue

                lbl = ttk.Label(lf, text=col + ":", font=("Verdana", 11, "bold"))
                lbl.grid(row=i, column=0, sticky="w", pady=4, padx=5)

                val = self.cadet_data.get(col, "")

                if col in self.label_categories:
                    widget = ttk.Combobox(
                        lf, values=self.label_categories[col],
                        state="readonly", font=("Verdana", 11))
                    widget.set(val if val in self.label_categories[col] else self.label_categories[col][0])
                    widget.grid(row=i, column=1, sticky="ew", pady=4, padx=5)
                else:
                    widget = tk.Entry(
                        lf, font=("Verdana", 11),
                        validate="key",
                        validatecommand=(validate_cmd, "%P", "%W")
                    )
                    widget.insert(0, str(val))
                    widget.grid(row=i, column=1, sticky="ew", pady=4, padx=5)

                    idx = self.feature_order.index(col)
                    limits = self._get_value_range_for_index(idx)
                    if limits:
                        min_val, max_val = limits
                        range_label = tk.Label(
                            lf, text=f"(от {min_val} до {max_val})",
                            font=("Verdana", 9), foreground="#555")
                        range_label.grid(row=i, column=2, sticky="w", padx=5)

                self.entries[col] = widget

            row += 1

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        save_btn = tk.Button(
            btn_frame, text="Сохранить", command=self._on_save,
            bg=AppConfig.palette["button_bg"], fg=AppConfig.palette["button_fg"],
            font=("Verdana", 12, "bold"), relief='raised', width=20, height=2
        )
        save_btn.pack()

    def _validate_and_color_field(self, proposed_value, widget_name):
        widget = self.nametowidget(widget_name)
        col_name = None
        for name, w in self.entries.items():
            if str(w) == widget_name:
                col_name = name
                break

        if proposed_value == "":
            widget.configure(bg="white")
            return True

        try:
            value = float(proposed_value.replace(",", "."))
            if col_name:
                idx = self.feature_order.index(col_name)
                limits = self._get_value_range_for_index(idx)
                if limits:
                    min_val, max_val = limits
                    if not (min_val <= value <= max_val):
                        widget.configure(bg="#ffdddd")  
                        return False

            widget.configure(bg="white")
            return True
        except ValueError:
            widget.configure(bg="#ffdddd")
            return False
        
    def _get_value_range_for_index(self, idx):
            ranges = {
                **dict.fromkeys(range(0, 2), (0, 15)),         
                **dict.fromkeys(range(2, 5), (0, 200)),        
                5: (0, 600),                                   
                6: (140, 215),                                 
                **dict.fromkeys(range(7, 9), (60, 120)),       
                9: (0, 15),                                    
                **dict.fromkeys(range(10, 13), (15, 180)),     
                **dict.fromkeys(range(13, 37), (0, 15)),       
                37: (0, 50),                                   
                38: (0, 30),                                   
                39: (0, 50),                                   
                **dict.fromkeys(range(41, 46), (0, 15)),       
            }
            return ranges.get(idx)

    def _group_fields(self):
        return {
            "Летная подготовка": [
                'контрольный налет (час)', 'Тренировочный.налет (час)\n',
                'Налет.ВСЭ(час)\n', 'Общий.налет(час)\n',
                'Полет.в.зону.на.ПП', 'Полет.по.марштуру.на.Нсред', 'Полет.2х180.(под шторкой)'
            ],
            "Когнитивные и поведенческие особенности": [
                'Уровень.перцептивн.модальности', 'Ур-нь.интел.лабильн.',
                'Продуктивность.(стэны)', 'Скорость.(стэны)', 'Точность.(стэны)',
                'Эффективность.(стэны)', 'Устойч.внимания.в.1.мин.(стэны)', 'Устойч.внимания.в.2.мин.(стэны)',
                'Самооценка(стэны)', 'Нейротизм.(стэны)', 'Экстраверсия.(стэны)', 'Открытость.опыту.(стэны)',
                'Согласие.(стэны)', 'Добросовестность.(стэны)', 'Атипичность.ответов.(стэны)', 'Экстраверсия.(стэны).1'
            ],
            "Физиология и здоровье": [
                'Рост (см)', 'Окруж.груд.клет.спокойно (см)', 'Окруж.груд.клет.макс.вдох (см)',
                'ДЖЕЛ.(л)', 'Проба.Вальсальвы.после(уд/30сек)', 'Проба.Штанге (сек.)', 'Проба.Генчи (сек.)'
            ],
            "Каналы восприятия": [
                'Аудиальный.канал.воспр.', 'Визуальный.канал.воспр', 'Кинестетический.канал.воспр.'
            ],
            "Психоэмоциональные шкалы": [
                'Тонус.(стэны)', 'Спокойствие.(стэны)', 'Эмоц.устойчивость.(стэны)',
                'Удовлетвор.жизнью.(стэны)', 'Удовлетвор.жизнью.(стэны).1'
            ],
            "Навыки": [
                'Командные.нав.(стэны)', 'Операторские.нав.(стэны)', 'Нав.связи.и.наблюд.(стэны)',
                'Водительские.нав.(стэны)', 'Нав.спец.назн.(стэны)', 'Технологические.нав.(стэны)'
            ],
            "Утомляемость": [
                'Утомление.(стэны)', 'Умств.утомление.(стэны)', 'Хрон.утомление.(стэны)'
            ],
            "Ошибки": [
                'Количество.ошибок', 'Время.обработки.(мин)', 'Количество.ошибок.1'
            ]
        }

    def _on_save(self):
        try:
            cadet_data = {}
            for col, widget in self.entries.items():
                val = widget.get().strip()
                if not val:
                    raise ValueError(f"Поле «{col}» не должно быть пустым")

                if col not in self.label_categories:
                    float(val.replace(",", "."))  # финальная проверка

                cadet_data[col] = val

            cadet_data["Уровень успешности в ЛП2"] = self.cadet_data.get("Уровень успешности в ЛП2", "не оценен")
            self.callback(cadet_data)
            self.destroy()
        except ValueError as ve:
            messagebox.showerror("Ошибка валидации", str(ve))
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


class AddCadetForm(StudentCardForm):
    def __init__(self, parent, feature_order, label_categories, callback):
        super().__init__(parent, feature_order, label_categories, callback)


class EditCadetForm(StudentCardForm):
    def __init__(self, parent, feature_order, label_categories, cadet_data, callback):
        super().__init__(parent, feature_order, label_categories, callback, cadet_data)
