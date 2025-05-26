from dataclasses import dataclass

@dataclass
class AppConfig:
    palette = {
        "bg": "#f5f7fa",
        "fg": "#2c3e50",
        "button_bg": "#3498db",
        "button_fg": "#ffffff",
        "highlight": "#aed6f1",
        "oddrow": "#f9f9f9",
        "evenrow": "#e8e8e8",
        "eligible": "#d4edda",
        "ineligible": "#f8d7da"
    }
    fonts = {
        "title": ("Verdana", 18, "bold"),
        "button": ("Arial", 12, "bold"),
        "label": ("Verdana", 10, "bold"),
        "text": ("Verdana", 10)
    }
    buttons = [
        ("Загрузить данные для обучения", "load_data"),
        ("Обучить модель", "train_model"),
        ("График обучения", "show_learning_curve"),
        ("Матрица ошибок", "show_confusion_matrix"),
        ("Гистограмма вклада признаков", "show_feature_importance"),
        ("Добавить курсанта вручную", "add_cadet_manually"),
        ("Добавить курсантов из файла", "add_cadets_from_file"),
        ("Показать добавленных курсантов", "show_cadets_table"),
        ("Экспорт в Excel", "export_to_excel")
    ]
