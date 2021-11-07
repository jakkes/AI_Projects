import ai.rl.utils.seed as seed


class ActorThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="ActorThread")

    def run(self):
        pass


class Actor(mp.Process):
    def __init__(self, daemon: bool = True):
        super().__init__(daemon=daemon, name="ActorProcess")

    def run(self):
        pass
