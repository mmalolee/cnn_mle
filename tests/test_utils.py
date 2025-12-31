# import pytest
# from src.config import BASE_DIR
# from src import utils
# import logging


# # --- FIXTURE -------------------------------------
# @pytest.fixture()
# def temporary_path(monkeypatch, tmp_path):
#     monkeypatch.setattr("src.utils.LOGS_DIR", tmp_path)

#     return tmp_path


# # --- TESTS ---------------------------------------
# def test_logs_dir_logic():
#     assert BASE_DIR in utils.LOGS_DIR.parents
#     assert utils.LOGS_DIR.is_absolute()


# def test_create_log_file(temporary_path):
#     log = utils.get_logger("file_test_logger")
#     log.info("file_test")
#     log_files = list(temporary_path.glob("*.log"))

#     assert len(log_files) > 0


# def test_logger_record_correct_message(caplog, temporary_path):
#     log = utils.get_logger("content_test")
#     test_message = "PYTEST MESSAGE FOR CONTENT"

#     with caplog.at_level(logging.INFO):
#         log.info(test_message)
#         log.error("ERROR during training")

#     assert test_message in caplog.text
#     assert "ERROR during training" in caplog.text

#     assert len(caplog.records) == 2
#     assert caplog.records[0].levelname == "INFO"
#     assert caplog.records[1].levelname == "ERROR"


# def test_logger_is_singleton():
#     log1 = utils.get_logger("logger")
#     log2 = utils.get_logger("logger")

#     assert log1 is log2


# def test_duplicate_handlers():
#     log1 = utils.get_logger("logger")
#     log1_handlers = len(log1.handlers)

#     log2 = utils.get_logger("logger")
#     log2_handlers = len(log2.handlers)

#     assert log1_handlers == log2_handlers
