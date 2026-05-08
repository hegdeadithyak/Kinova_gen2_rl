import threading
import time


class TUI:
    RST = "\033[0m";  BLD = "\033[1m";  DIM = "\033[2m"
    RED = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
    BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"
    WHT = "\033[97m"; GRY = "\033[90m"
    W = 70

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def _ts():
        return time.strftime("%H:%M:%S")

    def _top(self):
        return f"{self.BLD}{self.CYN}╔{'═' * (self.W - 2)}╗{self.RST}"

    def _bot(self):
        return f"{self.BLD}{self.CYN}╚{'═' * (self.W - 2)}╝{self.RST}"

    def _sep(self):
        return f"{self.BLD}{self.CYN}╠{'═' * (self.W - 2)}╣{self.RST}"

    def _row(self, text, color=""):
        inner = self.W - 2
        return (f"{self.BLD}{self.CYN}║{self.RST}"
                f"{color}{text:^{inner}}{self.RST}"
                f"{self.BLD}{self.CYN}║{self.RST}")

    def banner(self):
        print(f"\n{self._top()}")
        print(self._row("  CLICK-TO-FEED  ·  Kinova Jaco2  ·  Phased Joint Stepping  ",
                        f"{self.BLD}{self.WHT}"))
        print(self._sep())
        print(self._row("  SPACE = mouth   CLICK = target   Y = confirm   N = cancel   Q = quit  ",
                        self.GRY))
        print(self._bot())
        print()

    def separator(self):
        with self._lock:
            print(f"  {self.GRY}{'─' * (self.W - 4)}{self.RST}")

    def _emit(self, symbol, color, msg):
        with self._lock:
            print(f"  {self.GRY}{self._ts()}{self.RST}  {color}{symbol}{self.RST}  {msg}")

    def info(self, msg):
        self._emit("●", self.CYN, msg)

    def warn(self, msg):
        self._emit("▲", self.YLW, f"{self.YLW}{msg}{self.RST}")

    def error(self, msg):
        self._emit("✖", self.RED, f"{self.RED}{msg}{self.RST}")

    def success(self, msg):
        self._emit("✔", self.GRN, f"{self.BLD}{self.GRN}{msg}{self.RST}")

    def prompt(self, msg):
        with self._lock:
            print(f"\n  {self.BLD}{self.YLW}▶  {msg}{self.RST}\n")

    def phase_header(self, phase_num, name, detail=""):
        icons = {1: "①", 2: "②", 3: "③"}
        icon  = icons.get(phase_num, f"[{phase_num}]")
        label = f" {icon}  Phase {phase_num}  ·  {name} "
        pad   = max(2, (self.W - 4 - len(label)) // 2)
        with self._lock:
            print(f"\n  {self.BLD}{self.MAG}{'─' * pad}{label}{'─' * pad}{self.RST}")
            if detail:
                print(f"  {self.DIM}{self.GRY}  {detail}{self.RST}")

    def moving(self, label, elapsed, total):
        BAR    = 26
        frac   = min(elapsed / max(total, 0.001), 1.0)
        filled = int(BAR * frac)
        bar    = f"{self.GRN}{'█' * filled}{self.GRY}{'░' * (BAR - filled)}{self.RST}"
        remain = max(0.0, total - elapsed)
        with self._lock:
            print(f"\r    {self.CYN}{label:<10}{self.RST}[{bar}]  "
                  f"{self.GRY}{elapsed:.1f}s / {total:.1f}s  "
                  f"({self.WHT}{remain:.1f}s{self.GRY} left){self.RST}   ",
                  end="", flush=True)

    def phase_done(self, label):
        BAR = 26
        bar = f"{self.GRN}{'█' * BAR}{self.RST}"
        with self._lock:
            print(f"\r    {self.BLD}{self.GRN}{label:<10}{self.RST}"
                  f"[{bar}]  {self.BLD}{self.GRN}✔  complete{self.RST}" + " " * 20)

    def coord_block(self, label, x, y, z, color=""):
        c = color or self.GRY
        with self._lock:
            print(f"  {c}{label:<18}{self.RST}{self.WHT}x={x:+.3f}  y={y:+.3f}  z={z:+.3f}{self.RST}  m")

    def ready_prompt(self):
        self.separator()
        self.prompt("Ready — click the image or press SPACE to target the mouth")
        self.separator()
