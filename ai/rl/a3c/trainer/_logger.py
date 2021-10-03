import ai.utils.logging as logging


class Logger(logging.Server):
    def __init__(self):
        super().__init__(
            logging.field.Scalar("Episode/Reward"),
            logging.field.Scalar("Episode/Start value"),
            logging.field.Scalar("Agent/Loss"),
            logging.field.Scalar("Agent/Grad. magn."),
            name="a3c",
        )
