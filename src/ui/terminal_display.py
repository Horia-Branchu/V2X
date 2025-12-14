import sys
import shutil
import logging

logger = logging.getLogger("v2x")


class TerminalDisplay:
    def __init__(self, keys=None, logger_obj=None):
        self.keys = list(keys) if keys else []
        self._values = {k: "" for k in self.keys}
        self._initialized = False
        self._last_logged = {k: "" for k in self.keys}
        self.logger = logger_obj or logger

    def update(self, key, text):
        if key not in self._values:
            self.keys.append(key)
            self._values[key] = ""
            self._last_logged[key] = ""
        self._values[key] = text

    def render(self):
        if not sys.stdout.isatty():
            # Non-interactive: emit each changed value as an INFO
            for k in self.keys:
                v = self._values.get(k)
                if v and v != self._last_logged.get(k):
                    self.logger.info(v)
                    self._last_logged[k] = v
            return

        # Interactive terminal: print/update multiple lines in-place
        width = shutil.get_terminal_size((80, 24))[0]
        lines = [self._values.get(k, "") or "" for k in self.keys]

        if not self._initialized:
            for line in lines:
                sys.stdout.write(line.ljust(width) + "\n")
            sys.stdout.flush()
            self._initialized = True
            return

        # Move cursor up and overwrite lines
        up = f"\x1b[{len(lines)}A"
        sys.stdout.write(up)
        for line in lines:
            sys.stdout.write(line.ljust(width) + "\n")
        sys.stdout.flush()

    def finish(self):
        """End the interactive display: print a newline so following output
        starts below the display. Reset initialized state.
        """
        if sys.stdout.isatty() and self._initialized:
            # Move cursor to the line after the display
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._initialized = False


# module-level singleton used by env and features
terminal_display = TerminalDisplay(keys=["ENV"])