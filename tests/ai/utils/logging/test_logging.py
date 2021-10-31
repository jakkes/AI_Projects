import ai.utils.logging as logging


def test_logger():
    server = logging.Server(
        logging.field.Scalar("Test"),
        name="test"
    )
    port = server.start()
