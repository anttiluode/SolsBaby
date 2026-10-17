"""Desktop laboratory using the same simulator and probe as the benchmark."""
from dataclasses import replace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .experiment import EpisodeBatch, Protocol, expose, measure
from .model import Medium, Parameters
from .morphology import Genome
from .report import BACKGROUND, PANEL, TEXT, MUTED, COLORS, draw_tree, draw_probe, style_axes


class Laboratory:
    def __init__(self, root, seed=3):
        self.root, self.busy, self.last_branch = root, False, 0
        self.current_seed = seed
        self.seed = tk.IntVar(value=seed)
        self.active = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value='Teach a branch, then use the same probe to inspect its response.')
        self.protocol = Protocol(seeds=(seed,))
        self.protocol.validate()
        self.medium = Medium(Genome(seed=seed).grow())
        self.state = self.medium.initial()
        self.last_measurement = None
        root.title('SolsBaby · experience changes the response')
        root.geometry('1360x900')
        root.minsize(1020, 720)
        root.configure(background=BACKGROUND)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background=BACKGROUND)
        style.configure('TLabel', background=BACKGROUND, foreground=TEXT)
        style.configure('TButton', background=PANEL, foreground=TEXT, padding=(12, 10), borderwidth=0)
        style.map('TButton', background=[('active', '#29465a'), ('disabled', '#18212d')],
                  foreground=[('disabled', '#657487')])
        style.configure('TCheckbutton', background=BACKGROUND, foreground=TEXT)
        style.map('TCheckbutton', background=[('active', BACKGROUND)])
        style.configure('TSpinbox', fieldbackground=PANEL, foreground=TEXT, arrowcolor=TEXT)
        sidebar = ttk.Frame(root, padding=18)
        sidebar.pack(side='left', fill='y')
        ttk.Label(sidebar, text='SolsBaby', font=('Segoe UI', 24, 'bold')).pack(anchor='w', pady=(4, 4))
        ttk.Label(sidebar, text='Experience changes\nthe response.', font=('Segoe UI', 12),
                  foreground=MUTED).pack(anchor='w', pady=(0, 24))
        self.controls = []
        def button(text, command):
            widget = ttk.Button(sidebar, text=text, command=command)
            widget.pack(fill='x', pady=4)
            self.controls.append(widget)
            return widget
        ttk.Label(sidebar, text='1  TEACH', foreground=MUTED).pack(anchor='w', pady=(0, 6))
        for branch in range(4):
            button(f'Experience {chr(65+branch)}', lambda b=branch: self.teach(b))
        ttk.Label(sidebar, text='2  ASK THE MEDIUM', foreground=MUTED).pack(anchor='w', pady=(20, 6))
        button('Probe & compare', self.probe)
        button('Quick pulse · no writing', self.pulse)
        ttk.Label(sidebar, text='3  SEPARATE THE STATES', foreground=MUTED).pack(anchor='w', pady=(20, 6))
        button('Clear voltage', self.clear_fast)
        button('Erase learned EQ', self.erase)
        check = ttk.Checkbutton(sidebar, text='Active voltage boost', variable=self.active, command=self.switch_active)
        check.pack(anchor='w', pady=(20, 8))
        self.controls.append(check)
        row = ttk.Frame(sidebar)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text='Seed  ').pack(side='left')
        spinner = ttk.Spinbox(row, from_=0, to=999999, textvariable=self.seed, width=9)
        spinner.pack(side='left')
        self.controls.append(spinner)
        button('Regrow · resets memory', self.regrow)
        button('Save current figure…', self.save)
        ttk.Label(sidebar, text='Probe: voltage reset;\nwriting frozen; decay continues.\n\nDimensionless research model.\nNo GPU required.',
                  foreground=MUTED, font=('Segoe UI', 9), wraplength=220).pack(anchor='w', pady=(18, 0))
        body = ttk.Frame(root)
        body.pack(side='left', fill='both', expand=True)
        self.figure = Figure(figsize=(11, 8), facecolor=BACKGROUND, layout='constrained')
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, body, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill='x')
        ttk.Label(body, textvariable=self.status, foreground=MUTED, padding=10,
                  wraplength=1000).pack(fill='x')
        self.refresh()

    def refresh(self):
        m = self.medium.morphology
        ax = self.axes[0, 0]
        self.tree_image = draw_tree(ax, m, self.state.tag[0])
        ax.text(.01, .01, f"Voltage peak {np.max(np.abs(self.state.v)):.4f}  ·  slow p peak {self.state.tag.max():.3f}",
                transform=ax.transAxes, color=MUTED, fontsize=9)
        ax = self.axes[0, 1]
        ax.clear()
        if self.last_measurement:
            draw_probe(ax, *self.last_measurement, self.protocol)
        else:
            style_axes(ax)
            ax.set_title('Same probe · one somatic port')
            ax.text(.5, .5, 'Teach a branch, then click\nProbe & compare.', transform=ax.transAxes,
                    color=MUTED, ha='center', va='center', fontsize=13)
        ax = self.axes[1, 0]
        ax.clear()
        style_axes(ax)
        omega = np.geomspace(.01, 3, 80)
        before = self.medium.transfer(np.zeros(m.n), omega)
        after = self.medium.transfer(self.state.tag[0], omega)
        for b in range(4):
            ax.loglog(omega, abs(before[:, b]), '--', color=COLORS[b], alpha=.45)
            ax.loglog(omega, abs(after[:, b]), color=COLORS[b], label=chr(65+b))
        ax.set(xlabel='Angular frequency', ylabel='Tip-current → soma gain', title='Passive EQ · dashed = erased')
        ax.legend(facecolor=PANEL, edgecolor='#314157', labelcolor=TEXT, ncol=4, fontsize=9)
        ax = self.axes[1, 1]
        ax.clear()
        style_axes(ax)
        ax.bar(np.arange(m.n), self.state.tag[0], color=[MUTED if b < 0 else COLORS[b] for b in m.branch])
        ax.set(xlabel='Compartment', ylabel='Slow shunt p', ylim=(0, 1), title='Local memory · independent of voltage reset')
        boost = 'ON' if self.medium.parameters.active_strength else 'off'
        self.figure.suptitle(f'SolsBaby  /  seed {self.current_seed}  /  active boost {boost}',
                             color=TEXT, fontsize=16, fontweight='bold')
        self.canvas.draw_idle()

    def animate(self, frames, finished):
        self.busy = True
        for widget in self.controls:
            widget.configure(state='disabled')
        self.axes[0, 0].set_title('Voltage · replay of simulated dynamics', color=TEXT)
        selected = frames[::3]
        def tick(index=0):
            if index < len(selected):
                self.tree_image.set_array(selected[index])
                self.canvas.draw_idle()
                self.root.after(18, tick, index+1)
            else:
                self.busy = False
                for widget in self.controls:
                    widget.configure(state='normal')
                self.refresh()
                finished()
        tick()

    def teach(self, branch):
        self.last_branch = branch
        frames = []
        batch = EpisodeBatch(np.array([branch]), np.ones(1), np.zeros(1, dtype=int))
        self.state = expose(self.medium, batch, self.protocol, frames, initial_state=self.state)
        self.last_measurement = None
        self.status.set(f'Simulated experience {chr(65+branch)} and washout. Replaying its voltage dynamics…')
        self.animate(np.asarray(frames), lambda: self.status.set(
            f'Experience {chr(65+branch)} complete. Voltage is zero; slow shunt remains. Probe to read the changed response.'))

    def probe(self):
        learned = measure(self.medium, self.state, self.protocol, retain_frames=True)
        erased = measure(self.medium, self.state, self.protocol, erase=True)
        self.last_measurement = (learned, erased)
        ratio = np.linalg.norm(learned['trace']-erased['trace'])/max(np.linalg.norm(erased['trace']), 1e-15)
        self.status.set('Replaying the same branch-sweep probe. Writing is frozen during measurement.')
        self.animate(learned['frames'], lambda: self.status.set(
            f'Response difference from erased memory: {ratio:.1%}. Measurement used copies; the live memory is preserved.'))

    def pulse(self):
        g = np.zeros(self.medium.morphology.n)
        g[self.medium.morphology.tips[self.last_branch]] = .8
        frames = []
        for _ in range(15):
            self.medium.step(self.state, g, write=False)
            frames.append(self.state.v[0].copy())
        self.animate(np.asarray(frames), lambda: self.status.set(
            'A pulse changed voltage without writing memory. Clear voltage to see the states separate.'))

    def clear_fast(self):
        self.state.clear_fast()
        self.refresh()
        self.status.set('Voltage cleared. Learned conductance is unchanged.')

    def erase(self):
        self.state.erase_memory()
        self.last_measurement = None
        self.refresh()
        self.status.set('Learned EQ erased. Anatomy and current voltage are preserved.')

    def switch_active(self):
        self.medium = Medium(self.medium.morphology, replace(self.medium.parameters,
                             active_strength=1.2 if self.active.get() else 0.))
        self.last_measurement = None
        self.refresh()
        self.status.set('Active boost changed. The plotted EQ remains the frozen passive transfer function.')

    def regrow(self):
        try:
            seed = self.seed.get()
            morphology = Genome(seed=seed).grow()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror('Invalid seed', str(exc), parent=self.root)
            return
        self.protocol = Protocol(seeds=(seed,))
        self.current_seed = seed
        self.medium = Medium(morphology, Parameters(active_strength=1.2 if self.active.get() else 0.))
        self.state = self.medium.initial()
        self.last_measurement = None
        self.refresh()
        self.status.set('New developmental seed. Voltage and learned memory both start at zero.')

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.png',
                                          filetypes=[('PNG image', '*.png')], initialfile='solsbaby.png')
        if path:
            try:
                self.figure.savefig(path, dpi=160, facecolor=BACKGROUND)
                self.status.set(f'Saved {path}')
            except OSError as exc:
                messagebox.showerror('Could not save', str(exc), parent=self.root)


def launch(seed=3):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError('GUI needs a graphical display. Use: python -m solsbaby demo --out runs/demo') from exc
    try:
        Laboratory(root, seed)
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
