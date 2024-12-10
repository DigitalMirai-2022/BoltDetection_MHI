import tkinter as tk
from tkinter import font
import json

# tkinterのルートウィンドウ非表示
root = tk.Tk()
root.attributes("-topmost", True)
root.withdraw()


class QR_INFO:
    def __init__(self, parent1, qr_info):
        # ダイアログの作成
        self.parent = parent1  # 親ダイアログ
        self.qr_info = qr_info  # QR情報を保持
        self.dialog = tk.Toplevel(parent1)
        self.dialog.grab_set()  # このダイアログをフォーカスする
        self.run = False

        # ダイアログ表示
        self.plot_msgbox()

        # ダイアログをフレームの中央に表示
        self.dialog.update_idletasks()
        # 画面の幅と高さを取得
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        # ウィンドウサイズを取得
        window_width = self.dialog.winfo_width()
        window_height = self.dialog.winfo_height()
        # 中央に配置するためのX, Y座標を計算
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        # ウィンドウを指定位置に配置
        self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def plot_msgbox(self):
        self.dialog.geometry("450x120")  # 画面サイズの設定
        self.dialog.title("QR情報取得")  # 画面タイトルの設定

        # QR情報を表示するラベルの作成
        # フォントの設定（大きく、太字にする）
        label_font = font.Font(family="Helvetica", size=12, weight="bold")
        label = tk.Label(
            self.dialog,
            text=self.qr_info,
            wraplength=400,
            font=label_font,
        )
        label.pack(pady=20)

        # OK/キャンセルボタンを作成
        button_frame = tk.Frame(self.dialog)
        button_frame.pack()

        save_button = tk.Button(
            button_frame,
            text="OK",
            command=lambda: self.dialog_response(True),
            font=("ＭＳ Ｐゴシック", 12),
            width=10,  # 幅を設定
        )
        save_button.pack(side="left", padx=10)

        cancel_button = tk.Button(
            button_frame,
            text="キャンセル",
            command=lambda: self.dialog_response(False),
            font=("ＭＳ Ｐゴシック", 12),
            width=10,  # 幅を設定
        )
        cancel_button.pack(side="left", padx=10)

    def dialog_response(self, response):
        # OKまたはキャンセルに応じた処理
        if response:  # OKボタンが押された場合
            self.run = True
        else:  # キャンセルボタンが押された場合
            self.run = False
        self.dialog.destroy()


class msgbox_edit_config:
    def __init__(self, parent1, config_path, setting_value):
        self.parent = parent1  # 親ダイアログ
        self.config_path = config_path  # config.jsonファイルのパス
        self.dialog = tk.Toplevel(parent1)
        self.dialog.grab_set()  # このダイアログをフォーカスする
        self.entries = {}  # ユーザー入力欄を保持
        self.save = False

        # config.jsonから取得した
        self.config = setting_value

        # ダイアログ表示
        self.plot_msgbox()

        # ダイアログをフレームの中央に表示
        self.dialog.update_idletasks()
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        window_width = self.dialog.winfo_width()
        window_height = self.dialog.winfo_height()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def save_config(self):
        """config.jsonに入力値を書き込む"""
        try:
            # 入力された値を更新
            self.config["bolt status"]["loading interval"] = float(
                self.entries["判定間隔（秒）"].get()
            )
            self.config["bolt status"]["status change count"] = int(
                self.entries["クラス固定回数"].get()
            )
            self.config["Y range"]["Lower y"] = float(
                self.entries["ArUco垂直方向距離(mm)"].get()
            )
            # ファイルに保存
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            return self.config
        except ValueError as e:
            print(f"入力エラー: {e}")

    def plot_msgbox(self):
        self.dialog.geometry("300x180")  # 画面サイズの設定
        self.dialog.title("設定情報編集")  # 画面タイトルの設定

        # フォント設定
        label_font = font.Font(family="Helvetica", size=12, weight="bold")

        # 必要なフィールドのみ表示する
        fields = [
            ("判定間隔（秒）", self.config["bolt status"]["loading interval"]),
            (
                "クラス固定回数",
                self.config["bolt status"]["status change count"],
            ),
            ("ArUco垂直方向距離(mm)", self.config["Y range"]["Lower y"]),
        ]

        # 左揃えの配置を実現
        for key, value in fields:
            frame = tk.Frame(self.dialog)
            frame.pack(fill="x", padx=10, pady=5)

            # ラベル
            label = tk.Label(frame, text=f"{key}:", font=label_font, anchor="w")
            label.pack(side="left", padx=(0, 10))

            # エントリーフィールド
            entry = tk.Entry(frame, width=10, justify="right")
            entry.insert(0, value)  # 現在の値をデフォルトで挿入
            entry.pack(side="right", padx=(0, 0))  # 右揃え
            self.entries[key] = entry

        # OK/キャンセルボタンを作成
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)

        save_button = tk.Button(
            button_frame,
            text="OK",
            command=lambda: self.dialog_response(True),
            font=("ＭＳ Ｐゴシック", 12),
            width=10,
        )
        save_button.pack(side="left", padx=10)

        cancel_button = tk.Button(
            button_frame,
            text="キャンセル",
            command=lambda: self.dialog_response(False),
            font=("ＭＳ Ｐゴシック", 12),
            width=10,
        )
        cancel_button.pack(side="left", padx=10)

    def dialog_response(self, response):
        if response:  # 保存ボタンが押された場合
            self.config = self.save_config()
            self.save = True
        else:  # キャンセルボタンが押された場合
            self.save = False
        self.dialog.destroy()


# import tkinter as tk
# from tkinter import font
# import json
# from dataclasses import asdict  # Dataclass を辞書として保存
# from config_class import Config


# # class msgbox_edit_config:
# #     def __init__(self, parent1, config_path, config: Config):
# #         self.parent = parent1  # 親ダイアログ
# #         self.config_path = config_path  # config.jsonファイルのパス
# #         self.dialog = tk.Toplevel(parent1)
# #         self.dialog.grab_set()  # このダイアログをフォーカスする
# #         self.entries = {}  # ユーザー入力欄を保持
# #         self.save = False

# #         # Config dataclass から値を取得
# #         self.config = config

# #         # ダイアログ表示
# #         self.plot_msgbox()

# #         # ダイアログをフレームの中央に表示
# #         self.dialog.update_idletasks()
# #         screen_width = self.dialog.winfo_screenwidth()
# #         screen_height = self.dialog.winfo_screenheight()
# #         window_width = self.dialog.winfo_width()
# #         window_height = self.dialog.winfo_height()
# #         x = (screen_width // 2) - (window_width // 2)
# #         y = (screen_height // 2) - (window_height // 2)
# #         self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

# #     def save_config(self):
# #         """Config dataclass を更新し、JSONファイルに保存する"""
# #         try:
# #             # エントリーフィールドから値を取得して更新
# #             self.config.bolt_status.loading_interval = float(
# #                 self.entries["判定間隔（秒）"].get()
# #             )
# #             self.config.bolt_status.status_change_count = int(
# #                 self.entries["クラス固定回数"].get()
# #             )
# #             self.config.y_range.lower_y = float(
# #                 self.entries["ArUco垂直方向距離(mm)"].get()
# #             )

# #             # dataclass を辞書に変換して JSON に保存
# #             with open(self.config_path, "w") as f:
# #                 json.dump(asdict(self.config), f, indent=4)

# #             return self.config
# #         except ValueError as e:
# #             print(f"入力エラー: {e}")
# #             raise ValueError("Invalid input detected. Please check your entries.")

# #     def plot_msgbox(self):
# #         self.dialog.geometry("300x180")  # 画面サイズの設定
# #         self.dialog.title("設定情報編集")  # 画面タイトルの設定

# #         # フォント設定
# #         label_font = font.Font(family="Helvetica", size=12, weight="bold")

# #         # 必要なフィールドのみ表示する
# #         fields = [
# #             ("判定間隔（秒）", self.config.bolt_status.loading_interval),
# #             ("クラス固定回数", self.config.bolt_status.status_change_count),
# #             ("ArUco垂直方向距離(mm)", self.config.y_range.lower_y),
# #         ]

# #         for key, value in fields:
# #             frame = tk.Frame(self.dialog)
# #             frame.pack(fill="x", padx=10, pady=5)

# #             # ラベル
# #             label = tk.Label(frame, text=f"{key}:", font=label_font, anchor="w")
# #             label.pack(side="left", padx=(0, 10))

# #             # エントリーフィールド（右揃え）
# #             entry = tk.Entry(frame, width=10, justify="right")
# #             entry.insert(0, value)  # 現在の値をデフォルトで挿入
# #             entry.pack(side="right", padx=(0, 0))  # 右揃え
# #             self.entries[key] = entry

# #         # OK/キャンセルボタンを作成
# #         button_frame = tk.Frame(self.dialog)
# #         button_frame.pack(pady=20)

# #         save_button = tk.Button(
# #             button_frame,
# #             text="OK",
# #             command=lambda: self.dialog_response(True),
# #             font=("ＭＳ Ｐゴシック", 12),
# #             width=10,
# #         )
# #         save_button.pack(side="left", padx=10)

# #         cancel_button = tk.Button(
# #             button_frame,
# #             text="キャンセル",
# #             command=lambda: self.dialog_response(False),
# #             font=("ＭＳ Ｐゴシック", 12),
# #             width=10,
# #         )
# #         cancel_button.pack(side="left", padx=10)

# #     def dialog_response(self, response):
# #         if response:  # 保存ボタンが押された場合
# #             self.config = self.save_config()
# #             self.save = True
# #         else:  # キャンセルボタンが押された場合
# #             self.save = False
# #         self.dialog.destroy()


# # class msgbox_edit_config:
# #     def __init__(self, parent1, config_path, config: Config):
# #         self.parent = parent1  # 親ダイアログ
# #         self.config_path = config_path  # config.jsonファイルのパス
# #         self.dialog = tk.Toplevel(parent1)
# #         self.dialog.grab_set()  # このダイアログをフォーカスする
# #         self.entries = {}  # ユーザー入力欄を保持
# #         self.save = False

# #         # Config dataclass から値を取得
# #         self.config = config

# #         # ダイアログ表示
# #         self.plot_msgbox()

# #         # ダイアログをフレームの中央に表示
# #         self.dialog.update_idletasks()
# #         screen_width = self.dialog.winfo_screenwidth()
# #         screen_height = self.dialog.winfo_screenheight()
# #         window_width = self.dialog.winfo_width()
# #         window_height = self.dialog.winfo_height()
# #         x = (screen_width // 2) - (window_width // 2)
# #         y = (screen_height // 2) - (window_height // 2)
# #         self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

# #     def save_config(self):
# #         """Config dataclass を更新"""
# #         try:
# #             # エントリーフィールドから値を取得して更新
# #             self.config.bolt_status.loading_interval = float(
# #                 self.entries["判定間隔（秒）"].get()
# #             )
# #             self.config.bolt_status.status_change_count = int(
# #                 self.entries["クラス固定回数"].get()
# #             )
# #             self.config.y_range.lower_y = float(
# #                 self.entries["ArUco垂直方向距離(mm)"].get()
# #             )
# #             # 更新された設定をJSONファイルに保存
# #             with open(self.config_path, "w") as f:
# #                 json.dump(asdict(self.config), f, indent=4)

# #         except ValueError as e:
# #             print(f"入力エラー: {e}")
# #             raise ValueError("Invalid input detected. Please check your entries.")

# #     def plot_msgbox(self):
# #         self.dialog.geometry("300x180")  # 画面サイズの設定
# #         self.dialog.title("設定情報編集")  # 画面タイトルの設定

# #         # フォント設定
# #         label_font = font.Font(family="Helvetica", size=12, weight="bold")

# #         # 必要なフィールドのみ表示する
# #         fields = [
# #             ("判定間隔（秒）", self.config.bolt_status.loading_interval),
# #             ("クラス固定回数", self.config.bolt_status.status_change_count),
# #             ("ArUco垂直方向距離(mm)", self.config.y_range.lower_y),
# #         ]

# #         for key, value in fields:
# #             frame = tk.Frame(self.dialog)
# #             frame.pack(fill="x", padx=10, pady=5)

# #             # ラベル
# #             label = tk.Label(frame, text=f"{key}:", font=label_font, anchor="w")
# #             label.pack(side="left", padx=(0, 10))

# #             # エントリーフィールド（右揃え）
# #             entry = tk.Entry(frame, width=10, justify="right")
# #             entry.insert(0, value)  # 現在の値をデフォルトで挿入
# #             entry.pack(side="right", padx=(0, 0))  # 右揃え
# #             self.entries[key] = entry

# #         # OK/キャンセルボタンを作成
# #         button_frame = tk.Frame(self.dialog)
# #         button_frame.pack(pady=20)

# #         save_button = tk.Button(
# #             button_frame,
# #             text="OK",
# #             command=lambda: self.dialog_response(True),
# #             font=("ＭＳ Ｐゴシック", 12),
# #             width=10,
# #         )
# #         save_button.pack(side="left", padx=10)

# #         cancel_button = tk.Button(
# #             button_frame,
# #             text="キャンセル",
# #             command=lambda: self.dialog_response(False),
# #             font=("ＭＳ Ｐゴシック", 12),
# #             width=10,
# #         )
# #         cancel_button.pack(side="left", padx=10)

# #     def dialog_response(self, response):
# #         if response:  # 保存ボタンが押された場合
# #             self.save_config()  # Config を更新
# #             self.save = True
# #         else:  # キャンセルボタンが押された場合
# #             self.save = False
# #         self.dialog.destroy()
