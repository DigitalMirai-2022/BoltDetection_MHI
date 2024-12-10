from threading import Lock
import logging
import os


class LogMan:
    # ログ管理クラス

    # 条件1. 同じ型のインスタンスをprivate なクラス変数として定義する。
    #       1インスタンスしか生成しないことを保証するために利用する。
    _unique_instance = None
    _lock = Lock()  # クラスロック（マルチスレッド対応）

    # 条件2. コンストラクタの可視性をprivateとする。
    #        pythonの場合、コンストラクタをprivate定義できない。
    #        コンストラクタ呼び出しさせず、インスタンス取得をget_instanceに限定する。
    #        get_instanceからインスタンス取得を可能にするため、__init__は使用しない。
    #        初期化時に、__new__が__init__よりも先に呼び出される。
    def __new__(cls):
        raise NotImplementedError("Cannot initialize via Constructor")

    # インスタンス生成
    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    # 条件3:同じ型のインスタンスを返す `getInstance()` クラスメソッドを定義する。
    @classmethod
    def get_instance(cls):
        # インスタンス未生成の場合
        if not cls._unique_instance:
            with cls._lock:
                if not cls._unique_instance:
                    cls._unique_instance = cls.__internal_new__()
        return cls._unique_instance

    ############################################################################

    # フォーマット指定
    format = "[%(asctime)s] (%(levelname)s) %(message)s"

    # ロガー生成
    logger = logging.getLogger(os.path.basename(__file__))
    # エラーレベル設定
    logger.setLevel(20)  # DEBUG=10  INFO=20  WARINNG=30 ERROR=40 CRITICAL=50

    def debug(self, fn, message):
        """
        ログにデバッグメッセージを出力する DEBUG=10
        """
        print(message)
        message = fn + " : " + message
        self.logger.debug(message)

    def info(self, fn, message):
        """
        ログに情報メッセージを出力する INFO=20
        """
        print(message)
        message = fn + " : " + message
        self.logger.info(message)

    def warning(self, fn, message):
        """
        ログに警告メッセージを出力する  WARINNG=30
        """
        print(message)
        message = fn + " : " + message
        self.logger.warning(message)

    def error(self, fn, message):
        """
        ログにエラーメッセージを出力する  ERROR=40
        """
        print(message)
        message = fn + " : " + message
        self.logger.error(message)

    def critical(self, fn, message):
        """
        ログにクリティカルメッセージを出力する  CRITICAL=50
        """
        print(message)
        message = fn + " : " + message
        self.logger.critical(message)

    def exception(self, fn, message):
        """
        ログに例外処理のメッセージを付加して出力する
        """
        print(message)
        message = fn + " : " + message
        self.logger.exception(message)
