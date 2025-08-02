import os
import tkinter as tk
from tkinter import messagebox
import subprocess
from tkinter import font
import datetime

BASE_FOLDER = r"D:\Projects\OMR\new_abhigyan\Restructure"
# LOG_FILE_PATH = os.path.join(BASE_FOLDER, omr_template_name, date, "omr_process_log.txt")

# --- Custom font loading ---
FONT_PATH = os.path.join(os.path.dirname(__file__), "MyFont.ttf")  # change filename if needed
custom_font = None
if os.path.exists(FONT_PATH):
    try:
        custom_font = font.Font(file=FONT_PATH, size=11)  # custom size
    except:
        custom_font = None

class OMRInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Process")

        self.omr_template_name = tk.StringVar()
        self.date = tk.StringVar()
        self.select_all_var = tk.BooleanVar()

        # Bold font (same size)
        default_font = font.nametofont("TkDefaultFont")
        self.bold_font = font.Font(family=default_font.actual("family"),
                                   size=default_font.actual("size"),
                                   weight="bold")

        self.create_widgets()
        self.load_jobs()
        self.update_run_button_state()  # Initial disable

    def get_log_file_path(self):
        # Always save logs inside Images/<template_name>
        log_dir = os.path.join(BASE_FOLDER, "Images", self.omr_template_name.get())
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, "omr_process_log.txt")

    def write_log(self, template_name, date, batch, status):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp}\t{template_name}\t{date}\t{batch}\t{status}\n"
        
        log_file_path = self.get_log_file_path()  # <-- fixed here
        print(f"Log file path: {log_file_path}")  # <-- show in console

        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            self.log(f"⚠️ Failed to write log file: {e}")

    def create_widgets(self):
        tk.Label(self.root, text="Select Job (OMR Template):", font=self.bold_font).pack(anchor="w")
        self.job_listbox = tk.Listbox(self.root, height=5, exportselection=False)
        self.job_listbox.pack(fill=tk.X)
        self.job_listbox.bind("<<ListboxSelect>>", self.on_job_select)

        tk.Label(self.root, text="Select Date:", font=self.bold_font).pack(anchor="w")
        self.date_listbox = tk.Listbox(self.root, height=5, exportselection=False)
        self.date_listbox.pack(fill=tk.X)
        self.date_listbox.bind("<<ListboxSelect>>", self.on_date_select)

        tk.Label(self.root, text="Select Batch (Ctrl/Shift to multi-select):", font=self.bold_font).pack(anchor="w")
        self.batch_listbox = tk.Listbox(self.root, height=6, selectmode=tk.EXTENDED, exportselection=False)
        self.batch_listbox.pack(fill=tk.X)
        self.batch_listbox.bind("<<ListboxSelect>>", lambda e: self.update_run_button_state())

        self.select_all_checkbox = tk.Checkbutton(
            self.root, text="Select All Batches", variable=self.select_all_var,
            command=self.toggle_select_all
        )
        self.select_all_checkbox.pack(anchor="w", pady=2)

        self.run_button = tk.Button(
            self.root,
            text="Run Full Process",
            command=self.run_full_process,
            state=tk.DISABLED,
            width=18,    # slightly smaller than previous
            height=1     # slightly smaller height
        )
        if custom_font:  # apply custom font only to run button
            self.run_button.config(font=custom_font)
        self.run_button.pack(pady=10)

        tk.Label(self.root, text="Status:").pack(anchor="w")
        self.status_text = tk.Text(self.root, height=15, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()

    def load_jobs(self):
        images_path = os.path.join(BASE_FOLDER, "Images")
        jobs = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        self.job_listbox.delete(0, tk.END)
        for job in jobs:
            self.job_listbox.insert(tk.END, job)

    def on_job_select(self, event):
        selection = self.job_listbox.curselection()
        if not selection:
            return
        job_name = self.job_listbox.get(selection[0])
        self.omr_template_name.set(job_name)

        annotation_path = os.path.join(BASE_FOLDER, "Annotations", job_name)
        if not os.path.exists(annotation_path):
            messagebox.showerror("Error", f"Annotation unavailable for job {job_name}. Please annotate first.")
            return

        date_path = os.path.join(BASE_FOLDER, "Images", job_name)
        self.date_listbox.delete(0, tk.END)
        if os.path.exists(date_path):
            dates = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
            for d in dates:
                self.date_listbox.insert(tk.END, d)
        self.update_run_button_state()

    def on_date_select(self, event):
        selection = self.date_listbox.curselection()
        if not selection:
            return
        date_folder = self.date_listbox.get(selection[0])
        self.date.set(date_folder)

        batch_path = os.path.join(BASE_FOLDER, "Images", self.omr_template_name.get(), date_folder, "Input")
        self.batch_listbox.delete(0, tk.END)
        if not os.path.exists(batch_path):
            self.log(f"⚠️ No Input folder found for date {date_folder}")
            self.update_run_button_state()
            return

        batches = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        for b in batches:
            self.batch_listbox.insert(tk.END, b)
        self.update_run_button_state()

    def toggle_select_all(self):
        if self.select_all_var.get():
            self.batch_listbox.select_set(0, tk.END)
        else:
            self.batch_listbox.select_clear(0, tk.END)
        self.update_run_button_state()

    def get_selected_batches(self):
        selection = self.batch_listbox.curselection()
        return [self.batch_listbox.get(i) for i in selection]

    def update_run_button_state(self):
        job_selected = bool(self.omr_template_name.get())
        date_selected = bool(self.date.get())
        batches_selected = bool(self.get_selected_batches())

        if job_selected and date_selected and batches_selected:
            self.run_button.config(state=tk.NORMAL, bg="#77dd77", fg="black")  # pastel green
        else:
            self.run_button.config(state=tk.DISABLED, bg=self.root.cget("bg"), fg="black")

    def run_script(self, script_name, batch_name):
        code_base_path = os.path.join(BASE_FOLDER, "Codes_Integration")
        script_path = os.path.join(code_base_path, script_name)
        venv_python = os.path.join(BASE_FOLDER, "..", ".env", "Scripts", "python.exe")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        self.log(f"Running {script_name} for {batch_name} ...")
        result = subprocess.run(
            [venv_python, script_path,
             self.omr_template_name.get(),
             self.date.get(),
             batch_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env
        )
        self.log(result.stdout)
        if result.stderr:
            self.log("ERROR: " + result.stderr)
        if result.returncode != 0:
            raise Exception(f"{script_name} failed with exit code {result.returncode}")
        self.log(f"Finished {script_name} for {batch_name}\n")

    def run_full_process(self):
        self.run_button.config(state=tk.DISABLED, bg="#ff6961")  # pastel red while processing
        selected_batches = self.get_selected_batches()
        if not (self.omr_template_name.get() and self.date.get() and selected_batches):
            messagebox.showwarning("Missing Selection", "Please select Job, Date, and at least one Batch.")
            self.update_run_button_state()
            return

        try:
            for batch in selected_batches:
                try:
                    self.run_script("anchorDetection.py", batch)
                    self.run_script("fieldMapping.py", batch)
                    self.run_script("markedOption.py", batch)
                    self.run_script("runRequest.py", batch)
                    self.write_log(self.omr_template_name.get(), self.date.get(), batch, "SUCCESS")
                except Exception as e:
                    self.write_log(self.omr_template_name.get(), self.date.get(), batch, "FAIL")
                    raise e
            messagebox.showinfo("Success", "Full OMR Process Completed for selected batches!")
        except Exception as e:
            self.log(f"❌ Process aborted: {e}")
            messagebox.showerror("Execution Error", f"Process stopped!\n{e}")
        finally:
            self.update_run_button_state()


if __name__ == "__main__":
    root = tk.Tk()
    app = OMRInterface(root)
    root.mainloop()