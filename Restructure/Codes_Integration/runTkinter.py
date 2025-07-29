import os
import tkinter as tk
from tkinter import messagebox
import subprocess

BASE_FOLDER = r"D:\Projects\OMR\new_abhigyan\Restructure"
CODE_BASE_PATH = os.path.join(BASE_FOLDER, "Codes_Integration")

class OMRInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Process")

        self.omr_template_name = tk.StringVar()
        self.date = tk.StringVar()
        self.batch_name = tk.StringVar()

        self.create_widgets()
        self.load_jobs()

    def create_widgets(self):
        tk.Label(self.root, text="Select Job (OMR Template):").pack(anchor="w")
        self.job_listbox = tk.Listbox(self.root, height=5)
        self.job_listbox.pack(fill=tk.X)
        self.job_listbox.bind("<<ListboxSelect>>", self.on_job_select)

        tk.Label(self.root, text="Select Date:").pack(anchor="w")
        self.date_listbox = tk.Listbox(self.root, height=5)
        self.date_listbox.pack(fill=tk.X)
        self.date_listbox.bind("<<ListboxSelect>>", self.on_date_select)

        tk.Label(self.root, text="Select Batch:").pack(anchor="w")
        self.batch_listbox = tk.Listbox(self.root, height=5)
        self.batch_listbox.pack(fill=tk.X)
        self.batch_listbox.bind("<<ListboxSelect>>", self.on_batch_select)

        self.run_button = tk.Button(self.root, text="Run Full Process", command=self.run_full_process)
        self.run_button.pack(pady=10)

        # Status/Logs display
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
        if os.path.exists(date_path):
            dates = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
            self.date_listbox.delete(0, tk.END)
            for d in dates:
                self.date_listbox.insert(tk.END, d)

    def on_date_select(self, event):
        selection = self.date_listbox.curselection()
        if not selection:
            return
        date_folder = self.date_listbox.get(selection[0])
        self.date.set(date_folder)

        batch_path = os.path.join(BASE_FOLDER, "Images", self.omr_template_name.get(), date_folder, "Input")
        if not os.path.exists(batch_path):
            self.log(f"⚠️ No Input folder found for date {date_folder}")
            self.batch_listbox.delete(0, tk.END)
            return

        try:
            batches = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
            self.batch_listbox.delete(0, tk.END)
            for b in batches:
                self.batch_listbox.insert(tk.END, b)
        except Exception as e:
            self.log(f"Error fetching batches: {e}")

    def on_batch_select(self, event):
        selection = self.batch_listbox.curselection()
        if not selection:
            return
        self.batch_name.set(self.batch_listbox.get(selection[0]))
    
    def run_script(self, script_name):
        try:
            self.log(f"Running {script_name} ...")
            code_base_path = os.path.join(BASE_FOLDER, "Codes_Integration")
            script_path = os.path.join(code_base_path, script_name)

            # use venv python
            venv_python = os.path.join(BASE_FOLDER, "..", ".env", "Scripts", "python.exe")

            # ensure UTF-8 encoding for subprocess
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [venv_python, script_path,
                self.omr_template_name.get(),
                self.date.get(),
                self.batch_name.get()],
                capture_output=True,
                text=True,
                encoding="utf-8",    # <-- added
                env=env              # <-- added
            )

            self.log(result.stdout)
            if result.stderr:
                self.log("ERROR: " + result.stderr)
            if result.returncode != 0:
                raise Exception(f"{script_name} failed with exit code {result.returncode}")
            self.log(f"Finished {script_name}\n")
        except Exception as e:
            self.log(f"❌ {script_name} failed: {e}")
            raise

    def run_full_process(self):
        if not (self.omr_template_name.get() and self.date.get() and self.batch_name.get()):
            messagebox.showwarning("Missing Selection", "Please select Job, Date, and Batch.")
            return

        try:
            self.run_script("anchorDetection.py")
            self.run_script("fieldMapping.py")
            self.run_script("markedOption.py")
            # self.run_script("runRequest.py")
            messagebox.showinfo("Success", "Full OMR Process Completed!")
        except Exception as e:
            self.log(f"❌ Process aborted: {e}")
            messagebox.showerror("Execution Error", f"Process stopped!\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OMRInterface(root)
    root.mainloop()