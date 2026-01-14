import logging
import os
import sys
class ProgressBar:
    def __init__(self, logger: logging.Logger):
        self.total_trips = 0
        self.file_paths = []
        self.current = 0
        self.progress_colors_ansi = {
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RESET': '\033[0m'
        }
        self.logger = logger

    def load_trip_paths(self):
        for root, dirs, files in os.walk("config/", topdown=False):
            for name in files:
                if name.endswith(".trips.xml"):
                    self.file_paths.append(os.path.join(root, name))

    def count_total_trips(self):
        total = 0
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                for line in file:
                    if "<trip " in line:
                        total += 1
        self.total_trips =  total

    def update(self, step=1):
        self.current += step
        self.display()
    def return_progress_color(self, percent: float) -> str:
        if percent < 50:
            return self.progress_colors_ansi['RED']
        elif percent < 80:
            return self.progress_colors_ansi['YELLOW']
        else:
            return self.progress_colors_ansi['GREEN']
    def display_string(self, current: int=0, end: int = 1, info='', steps: bool = False):
        if steps:
            return self.display_string_bar(current, end, info=f"Steps: {current}/{end} {info}")
        else:
            return self.display_string_bar(current, self.total_trips, info=f"Arrived vehicles: {current}/{self.total_trips} {info}")
    def display_string_bar(self, current: int=0, end=1, info: str = ''):
        sys.stdout.write('\033[K') #clear to end of line (prevents artifact characters)
        percent = (current / end) * 100
        color = self.return_progress_color(percent)
        bar_length = 50
        filled_length = int(bar_length * current // end)
        filled_bar = f"{color}{'█' * filled_length}{self.progress_colors_ansi['RESET']}"
        empty_bar = '-' * (bar_length - filled_length)
        bar = filled_bar + empty_bar
        return f'|{bar}| {percent:.2f}%' + ' ' + info
