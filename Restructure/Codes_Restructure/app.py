import os
import tkinter as tk
from tkinter import messagebox, font
import subprocess
import datetime
import json

# --- Load config.json for BASE_FOLDER & VENV PYTHON ---
def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)

config = load_config()
BASE_FOLDER = config["base_folder"]
VENV_PYTHON = config["venv_folder"]  # <-- use venv python

# --- Optional Custom Font ---
FONT_PATH = os.path.join(os.path.dirname(__file__), "MyFont.ttf")
custom_font = None
if os.path.exists(FONT_PATH):
    try:
        custom_font = font.Font(file=FONT_PATH, size=11)
    except:
        custom_font = None


class OMRInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Process")

        # tkinter variables
        self.omr_template_name = tk.StringVar()
        self.date = tk.StringVar()
        self.select_all_var = tk.BooleanVar()

        # advanced flags (for hidden panel)
        self.save_anchor_images = tk.BooleanVar(value=False)
        self.save_mapped_images = tk.BooleanVar(value=False)
        self.draw_bboxes = tk.BooleanVar(value=False)
        self.full_mark_threshold_pct = tk.StringVar(value="")
        self.partial_mark_threshold_pct = tk.StringVar(value="")

        # store font reference
        self.custom_font = custom_font

        # Bold font
        default_font = font.nametofont("TkDefaultFont")
        self.bold_font = font.Font(family=default_font.actual("family"),
                                size=default_font.actual("size"),
                                weight="bold")

        self.show_advanced = tk.BooleanVar(value=False)
        
        self.create_widgets()
        self.load_jobs()
        self.update_run_button_state()

    def get_log_file_path(self):
        log_dir = os.path.join(BASE_FOLDER, "Images", self.omr_template_name.get())
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, "omr_process_log.txt")

    def write_log(self, template_name, date, batch, status):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp}\t{template_name}\t{date}\t{batch}\t{status}\n"
        log_file_path = self.get_log_file_path()
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            self.log(f"⚠️ Failed to write log file: {e}")

    def create_widgets(self):
        # --- Job Selection ---
        tk.Label(self.root, text="Select Job (OMR Template):", font=self.bold_font).pack(anchor="w")
        self.job_listbox = tk.Listbox(self.root, height=5, exportselection=False)
        self.job_listbox.pack(fill=tk.X)
        self.job_listbox.bind("<<ListboxSelect>>", self.on_job_select)

        # --- Date Selection ---
        tk.Label(self.root, text="Select Date:", font=self.bold_font).pack(anchor="w")
        self.date_listbox = tk.Listbox(self.root, height=5, exportselection=False)
        self.date_listbox.pack(fill=tk.X)
        self.date_listbox.bind("<<ListboxSelect>>", self.on_date_select)

        # --- Batch Selection ---
        tk.Label(self.root, text="Select Batch (Ctrl/Shift to multi-select):", font=self.bold_font).pack(anchor="w")
        self.batch_listbox = tk.Listbox(self.root, height=6, selectmode=tk.EXTENDED, exportselection=False)
        self.batch_listbox.pack(fill=tk.X)
        self.batch_listbox.bind("<<ListboxSelect>>", lambda e: self.update_run_button_state())

        # --- Select All ---
        self.select_all_checkbox = tk.Checkbutton(
            self.root, text="Select All Batches", variable=self.select_all_var,
            command=self.toggle_select_all
        )
        self.select_all_checkbox.pack(anchor="w", pady=2)

        # --- Run Button ---
        self.run_button = tk.Button(
            self.root, text="Run Full Process",
            command=self.run_full_process,
            state=tk.DISABLED,
            width=18, height=1
        )
        if self.custom_font:  # use custom font if loaded
            self.run_button.config(font=self.custom_font)
        self.run_button.pack(pady=5)

        # --- Advanced Settings Toggle ---
        self.advanced_btn = tk.Button(self.root, text="▶ Advanced Settings",
                                    command=self.toggle_advanced_panel)
        self.advanced_btn.pack(anchor="w")

        # --- Advanced Panel (hidden by default) ---
        self.advanced_panel = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)

        tk.Checkbutton(self.advanced_panel, text="Save Anchor Images",
                    variable=self.save_anchor_images).pack(anchor="w")
        tk.Checkbutton(self.advanced_panel, text="Save Mapped Images",
                    variable=self.save_mapped_images).pack(anchor="w")
        tk.Checkbutton(self.advanced_panel, text="Draw BBoxes",
                    variable=self.draw_bboxes).pack(anchor="w")

        tk.Label(self.advanced_panel, text="Full Mark Threshold (%)").pack(anchor="w")
        tk.Entry(self.advanced_panel, textvariable=self.full_mark_threshold_pct).pack(fill=tk.X)

        tk.Label(self.advanced_panel, text="Partial Mark Threshold (%)").pack(anchor="w")
        tk.Entry(self.advanced_panel, textvariable=self.partial_mark_threshold_pct).pack(fill=tk.X)

        # --- Status Area with Scrollbar ---
        tk.Label(self.root, text="Status:").pack(anchor="w")
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(status_frame, height=8, width=80, wrap=tk.WORD)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
    def toggle_advanced_panel(self):
        if self.show_advanced.get():
            self.advanced_panel.pack_forget()
            self.advanced_btn.config(text="▶ Advanced Settings")
            self.show_advanced.set(False)
        else:
            self.advanced_panel.pack(fill=tk.X, padx=5, pady=5)
            self.advanced_btn.config(text="▼ Advanced Settings")
            self.show_advanced.set(True)

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
            self.run_button.config(state=tk.NORMAL, bg="#77dd77", fg="black")
        else:
            self.run_button.config(state=tk.DISABLED, bg=self.root.cget("bg"), fg="black")

    def run_main_py(self, batch_name):
        self.log(f"Running main.py for {batch_name} ...")

        # Build argument list
        args = [VENV_PYTHON, "main.py",
                self.omr_template_name.get(),
                self.date.get(),
                batch_name]

        # Optional flags
        if self.save_anchor_images.get():
            args.append("--save-anchor")
        if self.save_mapped_images.get():
            args.append("--save-mapped")
        if self.draw_bboxes.get():
            args.append("--draw-bboxes")
        if self.full_mark_threshold_pct.get():
            args.extend(["--full-mark", str(self.full_mark_threshold_pct.get())])
        if self.partial_mark_threshold_pct.get():
            args.extend(["--partial-mark", str(self.partial_mark_threshold_pct.get())])

        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
        self.log(result.stdout)
        if result.stderr:
            self.log("ERROR: " + result.stderr)
        if result.returncode != 0:
            raise Exception(f"main.py failed with exit code {result.returncode}")
        self.log(f"Finished main.py for {batch_name}\n")


    def run_full_process(self):
        self.run_button.config(state=tk.DISABLED, bg="#ff6961")
        selected_batches = self.get_selected_batches()
        if not (self.omr_template_name.get() and self.date.get() and selected_batches):
            messagebox.showwarning("Missing Selection", "Please select Job, Date, and at least one Batch.")
            self.update_run_button_state()
            return

        try:
            for batch in selected_batches:
                try:
                    self.run_main_py(batch)
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