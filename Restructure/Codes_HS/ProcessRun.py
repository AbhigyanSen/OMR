import os
import tkinter as tk
from tkinter import messagebox
import subprocess

BASE_FOLDER = r"D:\Projects\OMR\new_abhigyan\Restructure"

class AnchorDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Process - Anchor Detection")
        self.root.geometry("600x400")

        self.job_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.batch_var = tk.StringVar()

        # Frame for job selection
        tk.Label(root, text="Select Job:").pack(anchor="w", padx=10, pady=(10, 0))
        self.job_listbox = tk.Listbox(root, height=5, exportselection=False)
        self.job_listbox.pack(fill="x", padx=10)
        self.job_listbox.bind("<<ListboxSelect>>", self.on_job_select)

        # Frame for date selection
        tk.Label(root, text="Select Date:").pack(anchor="w", padx=10, pady=(10, 0))
        self.date_listbox = tk.Listbox(root, height=5, exportselection=False)
        self.date_listbox.pack(fill="x", padx=10)
        self.date_listbox.bind("<<ListboxSelect>>", self.on_date_select)

        # Frame for batch selection
        tk.Label(root, text="Select Batch:").pack(anchor="w", padx=10, pady=(10, 0))
        self.batch_listbox = tk.Listbox(root, height=5, exportselection=False)
        self.batch_listbox.pack(fill="x", padx=10)

        # Run Button
        tk.Button(root, text="Run Anchor Detection", command=self.run_anchor_detection).pack(pady=20)

        self.load_jobs()

    def load_jobs(self):
        images_path = os.path.join(BASE_FOLDER, "Images")
        if not os.path.exists(images_path):
            messagebox.showerror("Error", "Images folder not found in base path")
            return

        jobs = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        self.job_listbox.delete(0, tk.END)
        for job in jobs:
            self.job_listbox.insert(tk.END, job)

    def on_job_select(self, event):
        selection = self.job_listbox.curselection()
        if not selection:
            return
        job = self.job_listbox.get(selection[0])

        # Check annotation availability
        annotations_path = os.path.join(BASE_FOLDER, "Annotations", job)
        if not os.path.exists(annotations_path):
            messagebox.showwarning("Annotation Missing", f"Annotation unavailable for job '{job}'. Please annotate first.")
            return

        self.job_var.set(job)
        # Load date folders
        date_path = os.path.join(BASE_FOLDER, "Images", job)
        dates = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        self.date_listbox.delete(0, tk.END)
        for date in dates:
            self.date_listbox.insert(tk.END, date)
        self.batch_listbox.delete(0, tk.END)

    def on_date_select(self, event):
        selection = self.date_listbox.curselection()
        if not selection:
            return
        date = self.date_listbox.get(selection[0])
        self.date_var.set(date)

        batch_path = os.path.join(BASE_FOLDER, "Images", self.job_var.get(), date, "Input")
        self.batch_listbox.delete(0, tk.END)

        if not os.path.exists(batch_path):
            messagebox.showwarning("No Input Folder", f"No 'Input' folder found for date: {date}")
            return

        batches = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
        if not batches:
            messagebox.showwarning("No Batches Found", f"No batch folders found in:\n{batch_path}")
            return

        for batch in batches:
            self.batch_listbox.insert(tk.END, batch)

    def run_anchor_detection(self):
        selection = self.batch_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select a batch")
            return

        batch = self.batch_listbox.get(selection[0])
        self.batch_var.set(batch)

        # Final parameters
        omr_template_name = self.job_var.get()
        date = self.date_var.get()
        batch_name = self.batch_var.get()

        if not omr_template_name or not date or not batch_name:
            messagebox.showwarning("Missing Selection", "Please select job, date and batch")
            return

        # Call anchorDetection.py
        command = [
            "python",
            "anchorDetection.py",
            "--template", omr_template_name,
            "--date", date,
            "--batch", batch_name
        ]
        try:
            subprocess.run(command, check=True)
            messagebox.showinfo("Success", "Anchor Detection completed successfully.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Anchor Detection failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnchorDetectionUI(root)
    root.mainloop()
