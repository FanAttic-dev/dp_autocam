import time


class Stopwatch:
    def __init__(self, name):
        self.name = name
        self.time_start = None
        self.elapsed_sec = None

    @property
    def is_stopped(self):
        return self.elapsed_sec is not None

    def start(self):
        self.time_start = time.time()
        return self

    def stop(self):
        assert self.time_start is not None
        self.time_end = time.time()
        self.elapsed_sec = self.time_end - self.time_start


class Profiler:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.stopwatches = {}

    def start(self, name):
        assert (name not in self.stopwatches)
        self.stopwatches[name] = Stopwatch(name).start()

    def stop(self, name):
        assert name in self.stopwatches
        self.stopwatches[name].stop()

    def print_summary(self):
        txt = f"""[Frame {self.frame_id:04d}] """

        stopwatches = [f"{k}: {v.elapsed_sec*1000:3.0f}ms" for k,
                       v in self.stopwatches.items() if v.is_stopped and k != "Total"]
        txt += " | ".join(stopwatches)

        total_elapsed_sec = self.stopwatches["Total"].elapsed_sec
        txt += f" || Total: {total_elapsed_sec:.2f}s ({1/total_elapsed_sec:.1f}fps)\n"
        print(txt)
