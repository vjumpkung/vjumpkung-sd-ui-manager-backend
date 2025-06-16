import asyncio
import json
import os
from datetime import datetime

from config.load_config import PROGRAM_LOG, UI_TYPE
from event_handler import manager
from worker.create_log_file import touch_files

touch_files()


class ProgramLog:

    log_path = ""

    _log_lst = []

    key = ""

    def __init__(self, PROGRAM_LOG, KEY):

        self.log_path = PROGRAM_LOG
        self.key = KEY

        self._log_lst.clear()
        with open(self.log_path, "r", encoding="utf-8") as fp:
            f = fp.readlines()
            for data in f:
                entry = {"t": datetime.now().isoformat(), "m": data.strip()}
                self._log_lst.append(entry)

    def get(self):
        return self._log_lst

    async def monitor_log(self):
        # Get initial file size
        log_file_path = self.log_path
        file_size = os.path.getsize(log_file_path)
        try:
            stop_var = False
            with open(log_file_path, "r", encoding="utf-8") as f:
                # Move to the end of the file
                f.seek(file_size)

                # Continue monitoring for changes
                while True:
                    try:
                        # Check if file size has changed
                        current_size = os.path.getsize(log_file_path)

                        if current_size > file_size:
                            # Read only the new data
                            f.seek(file_size)
                            new_data = f.read()
                            # print(new_data, end="", flush=True)

                            s = {
                                "key": self.key,
                                "type": "logs",
                                "data": {
                                    "m": new_data.strip(),
                                },
                            }

                            entry = {
                                "t": datetime.now().isoformat(),
                                "m": new_data.strip(),
                            }
                            self._log_lst.append(entry)

                            await manager.broadcast(json.dumps(s))

                            file_size = current_size

                        # If file has been truncated (rotated), start from beginning
                        elif current_size < file_size:
                            f.seek(0)
                            new_data = f.read()
                            s = {
                                "type": "logs",
                                "data": {
                                    "m": new_data.strip(),
                                },
                            }

                            entry = {
                                "t": datetime.now().isoformat(),
                                "m": new_data.strip(),
                            }
                            self._log_lst.append(entry)

                            await manager.broadcast(json.dumps(s))
                            file_size = current_size

                        # time.sleep(0.1)
                        await asyncio.sleep(0.1)
                    except KeyboardInterrupt as e:
                        stop_var = True
                        break

            print("\nLog monitoring stopped.")

        except Exception as e:
            print(f"\nError monitoring log file: {e}")
            return


programLog = ProgramLog(PROGRAM_LOG, UI_TYPE)
