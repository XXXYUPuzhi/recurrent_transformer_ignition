"""
generate_pdf_report.py — PDF lab report with embedded figures

Generates a formatted PDF report using fpdf2, embedding all result
figures directly into the document.

Author: Puzhi YU
Date:   January 2026
"""

import os
from fpdf import FPDF

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, 'results')


class LabReport(FPDF):
    """Custom PDF class with header/footer and helper methods."""

    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=20)
        # Register fonts - use built-in fonts only for maximum compatibility
        self.title_text = ''

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, 'Ignition Mechanisms in Information-Bottleneck Recurrent Networks', align='C')
            self.ln(3)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    # ── Helper methods ────────────────────────────

    def section_title(self, num, title):
        """Major section heading (e.g., '2. Methods')"""
        self.ln(4)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, f'{num}. {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(44, 62, 80)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, num, title):
        """Subsection heading (e.g., '2.1 Task Design')"""
        self.ln(2)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(44, 62, 80)
        self.cell(0, 7, f'{num} {title}', new_x='LMARGIN', new_y='NEXT')
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        """Normal paragraph text."""
        self.set_font('Times', '', 10.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        """Bold paragraph text."""
        self.set_font('Times', 'B', 10.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def italic_text(self, text):
        """Italic paragraph text."""
        self.set_font('Times', 'I', 10.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def blockquote(self, text):
        """Indented blockquote with left border."""
        x = self.get_x()
        y = self.get_y()
        self.set_fill_color(240, 247, 253)
        self.set_draw_color(52, 152, 219)
        # Estimate height
        self.set_font('Times', 'I', 10)
        # Draw background + border
        save_x = self.get_x()
        self.set_x(15)
        self.set_font('Times', 'I', 10)
        self.multi_cell(175, 5, text, border=0, fill=False)
        end_y = self.get_y()
        # Draw left border line
        self.set_draw_color(52, 152, 219)
        self.set_line_width(0.8)
        self.line(13, y, 13, end_y)
        self.set_line_width(0.2)
        self.ln(2)

    def add_figure(self, img_path, caption, fig_id='', width=170):
        """Add an image with caption, centered."""
        if not os.path.exists(img_path):
            self.body_text(f'[Image not found: {img_path}]')
            return

        # Check if we need a new page (estimate ~60mm for image + caption)
        if self.get_y() > 220:
            self.add_page()

        x_offset = (210 - width) / 2
        self.image(img_path, x=x_offset, w=width)
        self.ln(2)
        # Caption
        self.set_font('Helvetica', '', 8.5)
        self.set_text_color(80, 80, 80)
        label = f'{fig_id}: ' if fig_id else ''
        self.set_x(15)
        self.multi_cell(180, 4, f'{label}{caption}', align='C')
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None, title=''):
        """Add a formatted table."""
        if title:
            self.set_font('Helvetica', 'B', 9.5)
            self.cell(0, 6, title, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)

        n_cols = len(headers)
        if col_widths is None:
            total_w = 190
            col_widths = [total_w / n_cols] * n_cols

        # Check if enough space
        estimated_h = (len(rows) + 1) * 6 + 5
        if self.get_y() + estimated_h > 275:
            self.add_page()

        # Header
        self.set_font('Helvetica', 'B', 8.5)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, align='C', fill=True)
        self.ln()

        # Rows
        self.set_font('Times', '', 8.5)
        self.set_text_color(0, 0, 0)
        for r_idx, row in enumerate(rows):
            is_highlight = row[0].startswith('*')
            if is_highlight:
                self.set_fill_color(234, 244, 253)
                row = [c.lstrip('*') for c in row]
            else:
                self.set_fill_color(248, 249, 250) if r_idx % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, c in enumerate(row):
                if is_highlight:
                    self.set_font('Times', 'B', 8.5)
                else:
                    self.set_font('Times', '', 8.5)
                self.cell(col_widths[i], 5.5, c, border=1, align='C', fill=True)
            self.ln()
        self.ln(3)

    def bullet(self, text, indent=15):
        """Single bullet point."""
        self.set_font('Times', '', 10.5)
        self.set_x(indent)
        self.cell(5, 5, chr(8226))  # bullet char
        self.multi_cell(170, 5, text)


# ══════════════════════════════════════════════════
# BUILD THE REPORT
# ══════════════════════════════════════════════════

pdf = LabReport()
pdf.alias_nb_pages()
pdf.set_display_mode('fullpage')

# ── Title page ────────────────────────────────────
pdf.add_page()
pdf.ln(30)
pdf.set_font('Helvetica', 'B', 16)
pdf.set_text_color(44, 62, 80)
pdf.multi_cell(0, 9, 'Ignition Mechanisms in\nInformation-Bottleneck Recurrent Networks:\nA Computational Study on\nOctagon Geometric Sequence Tasks', align='C')
pdf.ln(10)

pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 7, 'Puzhi Yu', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)
pdf.set_font('Helvetica', '', 10)
pdf.cell(0, 6, 'February 2026', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 6, 'Repository: causal_emergence/', align='C', new_x='LMARGIN', new_y='NEXT')

pdf.ln(12)
pdf.set_draw_color(44, 62, 80)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(8)

# Abstract
pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(44, 62, 80)
pdf.cell(0, 6, 'Abstract', new_x='LMARGIN', new_y='NEXT')
pdf.set_text_color(0, 0, 0)
pdf.ln(2)
pdf.set_font('Times', 'I', 10)
pdf.multi_cell(0, 4.8,
    'We construct a minimalist Recurrent Transformer augmented with a Variational Information '
    'Bottleneck (VIB) and investigate the computational mechanisms underlying the "Ignition" '
    'phenomenon from Dehaene\'s Global Neuronal Workspace (GNW) theory. Using an octagon geometric '
    'sequence prediction task with three difficulty levels (Simple, Nested, Random), we demonstrate: '
    '(1) A 13,292-parameter weight-sharing recurrent network spontaneously learns nested sequence '
    'rules from raw geometric coordinates (90-100% accuracy vs. 12.5% chance). '
    '(2) Information bottleneck pressure (beta > 0) compresses internal representations by ~85% '
    '(KL: 13 -> 2) without accuracy loss, while beta=0 shows KL explosion to 200+. '
    '(3) Linear probing reveals rule information gradually emerges across recurrent steps '
    '(33% at t=0 to 91% at t=6), supporting "Recurrence as Thinking Time." '
    '(4) The 2D bottleneck spontaneously forms separated vertex clusters with circular topology, '
    'yet encodes position nonlinearly and discards rule-type information -- revealing distributed '
    'computation. We additionally document: VIB inside the recurrent loop causes Posterior Collapse; '
    'relocating it to the readout stage aligns with GNW\'s "broadcast" semantics.')
pdf.ln(3)
pdf.set_font('Times', '', 9)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 5, 'Keywords: Information Bottleneck, Global Neuronal Workspace, Ignition, Recurrent Transformer, Representation Learning',
         new_x='LMARGIN', new_y='NEXT')

# ══════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('1', 'Introduction')

pdf.subsection_title('1.1', 'Background: Global Neuronal Workspace Theory and Ignition')
pdf.body_text(
    'Stanislas Dehaene\'s Global Neuronal Workspace (GNW) theory [1] posits that conscious '
    'information processing corresponds to a special neural event -- Ignition: when sensory '
    'information exceeds a threshold, it triggers a nonlinear, cross-cortical spreading activation '
    'that rapidly propagates across the brain, forming a stable global representation. In EEG/MEG '
    'experiments, this manifests as the sudden onset of the P3 wave and synchronized firing across '
    'fronto-parietal regions.')
pdf.body_text(
    'However, the computational mechanisms behind Ignition remain unclear. Specifically: '
    'What information processing operations correspond to "discarding details and distilling rules"? '
    'What role does recurrence play in rule extraction? '
    'Is information compression a necessary condition or consequence of "understanding"?')

pdf.subsection_title('1.2', 'Information Bottleneck Framework')
pdf.body_text(
    'The Information Bottleneck (IB) principle [2] provides a formal framework: given input X and '
    'target Y, find bottleneck variable Z that maximizes I(Z;Y) while minimizing I(X;Z). Alemi et '
    'al. [3] extended this to the Variational Information Bottleneck (VIB), enabling end-to-end '
    'training. This study combines VIB with a recurrent Transformer to explore how information '
    'bottleneck facilitates rule emergence in sequential learning.')

pdf.subsection_title('1.3', 'Research Questions')
pdf.blockquote('Q1: Can a minimalist recurrent network (single layer, weight-sharing, 13K parameters) '
               'spontaneously learn nested structural rules from geometric coordinate sequences?')
pdf.blockquote('Q2: Does information bottleneck pressure (VIB, parameter beta) produce more compact '
               'and stable internal representations without sacrificing prediction accuracy?')
pdf.blockquote('Q3: Does rule information emerge gradually across recurrent steps, or appear instantaneously?')

# ══════════════════════════════════════════════════
# 2. METHODS
# ══════════════════════════════════════════════════
pdf.section_title('2', 'Methods')

pdf.subsection_title('2.1', 'Experimental Paradigm: Octagon Sequence Task')
pdf.body_text(
    'We computationally implement a variant of the classical paradigm from Dehaene\'s laboratory, '
    'using vertex sequences on a regular octagon as stimuli. Eight vertices (k = 0,...,7) are encoded '
    'as geometric coordinates on the unit circle: input_k = [cos(2pi*k/8), sin(2pi*k/8)]. '
    'The task is autoregressive next-step prediction (8-class, chance = 12.5%).')

# Paradigm figure
pdf.add_figure(os.path.join(R, 'paradigm_schematic.png'),
    '(a) Simple: clockwise +1 step per time point (period 8). '
    '(b) Nested: alternating +2 (forward) and -1 (backward) steps ("two steps forward, one step back"; '
    'micro-period 2, macro-period 16). (c) Random (control): uniformly random jumps, incompressible.',
    fig_id='Figure 1', width=175)

pdf.add_table(
    ['Type', 'Generation Rule', 'Example', 'Complexity'],
    [['Simple', 'x(t+1) = (x(t)+1) mod 8', '0->1->2->3->4->5...', 'Minimal'],
     ['Nested', 'Even: +2; Odd: -1 (mod 8)', '0->2->1->3->2->4...', 'Medium-high'],
     ['Random', 'x(t+1) ~ U{0,...,7}', '0->5->2->7->1...', 'Incompressible']],
    col_widths=[22, 52, 52, 30])

pdf.body_text('Dataset: 10,000 sequences per type (30,000 total), L=10, split 70/15/15 (train/val/test), seed=42.')

pdf.subsection_title('2.2', 'Model Architecture')
pdf.body_text(
    'The model follows a minimalist design: a single Transformer encoder layer is reused across '
    'T=8 recurrent steps (weight-sharing), forcing iterative refinement rather than memorization. '
    'A Variational Information Bottleneck (VIB) is applied ONCE at the readout stage (after all '
    'recurrent loops), compressing the 32-dim hidden state into a 2-dim bottleneck via the '
    'reparameterization trick (z = mu + eps*sigma). The output head then decodes z back to 32-dim '
    'and predicts the next vertex (8-class).')

pdf.add_table(
    ['Component', 'Details', 'Parameters'],
    [['Input embedding', '2 -> d_model=32', '64'],
     ['Transformer layer', 'd=32, nhead=4, FFN=128, Pre-LN', '~12,800'],
     ['VIB layer', '32 -> 2 -> 32 (mu + logvar + decoder)', '192'],
     ['Output head', '32 -> 8 classes', '264'],
     ['*Total', '*', '*13,292']],
    col_widths=[45, 95, 30])

pdf.subsection_title('2.3', 'Loss Function and Training')
pdf.body_text(
    'Loss = CrossEntropy(logits, y) + beta(t) * KL(N(mu,sigma^2) || N(0,I)). '
    'KL has analytical form: -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2). '
    'Beta annealing: beta(t) = beta_max * tanh(t / T_warmup), T_warmup = 0.5 * T_total. '
    'Optimizer: AdamW (lr=1e-3, weight_decay=1e-4), grad clip max_norm=1.0, batch=128.')

pdf.subsection_title('2.4', 'Evaluation Metrics')
pdf.body_text(
    'Three Ignition monitoring metrics recorded every 100 steps on validation set, per type: '
    '(1) Accuracy -- behavioral performance; '
    '(2) KL Divergence -- information compression; '
    '(3) RCR (||H_t - H_{t-1}||) -- attractor stability.')

pdf.subsection_title('2.5', 'Ablation Study Design')
pdf.add_table(
    ['Run', 'T', 'beta_max', 'Steps', 'Role'],
    [['T8_b00', '8', '0.0', '1500', 'Ablation B: recurrence, no bottleneck'],
     ['T8_b001', '8', '0.01', '1500', 'Weak bottleneck'],
     ['*main_b01', '*8', '*0.1', '*3000', '*Primary model'],
     ['T8_b01', '8', '0.1', '1500', 'Beta sweep'],
     ['T8_b10', '8', '1.0', '1500', 'Strong bottleneck'],
     ['T1_b01', '1', '0.1', '1500', 'Ablation A: bottleneck, no recurrence'],
     ['T1_b0', '1', '0.0', '1500', 'Ablation C: feedforward baseline']],
    col_widths=[24, 10, 18, 16, 80])

# ══════════════════════════════════════════════════
# 3. RESULTS
# ══════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('3', 'Results')

pdf.subsection_title('3.1', 'Main Model Learning Dynamics')

pdf.add_figure(os.path.join(R, 'ignition_main_b01.png'),
    'Three-panel Ignition curve for the main model (T=8, beta=0.1, 3000 steps). '
    'Top: Accuracy for Simple (red), Nested (blue), Random (gray). Ignition at step 100. '
    'Middle: KL Divergence decreases from ~13.7 to ~2.0 (85% compression). '
    'Bottom: RCR decreases from ~3.1 to ~2.3 (attractor convergence).',
    fig_id='Figure 2', width=120)

pdf.bold_text('Accuracy (Top):')
pdf.body_text(
    'Within ~100 steps (3.3s on CPU), Simple and Nested accuracy jump to 92.6% and 70.4%. '
    'By step 200, both stabilize at 93-96%. Random stays at 12.0-12.9% throughout, ruling out overfitting.')

pdf.bold_text('KL Divergence (Middle):')
pdf.body_text(
    'Initial KL ~ 13.7 monotonically decreases to ~ 2.0, an overall reduction of approximately 85%. '
    'While prediction accuracy remains constant, the internal representation becomes increasingly compact.')

pdf.bold_text('RCR (Bottom):')
pdf.body_text(
    'Decreases from ~3.1 to ~2.3, paralleling the KL decrease. As rule extraction completes, '
    'per-step state changes shrink -- the system approaches a stable attractor.')

# ── 3.2 Beta sweep ────────────────────────────────
pdf.subsection_title('3.2', 'Effect of Information Bottleneck Pressure (Beta Sweep)')

pdf.add_figure(os.path.join(R, 'beta_sweep_nested.png'),
    'Comparison of four beta values (0.0, 0.01, 0.1, 1.0) for T=8 on Nested sequences. '
    'Left: All configs reach 90-100% accuracy. '
    'Middle: CRITICAL -- beta=0 (gray) KL explodes to ~150; beta=0.1 (blue) compresses to ~2; beta=1.0 to ~0.4. '
    'Right: beta=0 RCR rises continuously; beta>0 converges.',
    fig_id='Figure 3', width=175)

pdf.add_figure(os.path.join(R, 'beta_sweep_simple.png'),
    'Same beta sweep on Simple sequences. KL and RCR patterns consistent with Nested (Figure 3).',
    fig_id='Figure 4', width=175)

pdf.blockquote(
    'Key Finding: Under equivalent accuracy, VIB (beta > 0) fundamentally alters the model\'s '
    'internal state -- from "inflating" to "compressing," from "turbulent" to "stable." '
    'This constitutes a dual Ignition signal of information compression + attractor dynamics.')

# ── 3.3 Ablation ──────────────────────────────────
pdf.subsection_title('3.3', '2x2 Ablation: Disentangling Recurrence and Bottleneck')

pdf.add_figure(os.path.join(R, 'ablation_2x2_nested.png'),
    'Ablation comparison on Nested sequences. Blue: Main (T=8, beta=0.1). Red: Ablation A (T=1, beta=0.1). '
    'Purple: Ablation C (T=1, beta=0). RCR panel (right) shows the most dramatic difference.',
    fig_id='Figure 5', width=175)

pdf.bold_text('Table 1: Final metrics for all experimental runs.')
pdf.add_table(
    ['Run', 'T', 'beta', 'Simple', 'Nested', 'Random', 'KL', 'RCR'],
    [['T8_b00', '8', '0.0', '93.7%', '94.8%', '13.1%', '208.6', '5.39'],
     ['T8_b001', '8', '0.01', '90.0%', '100.0%', '12.2%', '5.84', '2.79'],
     ['*main_b01', '*8', '*0.1', '*93.8%', '*96.3%', '*12.6%', '*2.01', '*2.38'],
     ['T8_b01', '8', '0.1', '95.0%', '95.0%', '12.4%', '2.27', '2.42'],
     ['T8_b10', '8', '1.0', '95.0%', '95.0%', '12.4%', '0.36', '2.03'],
     ['T1_b01', '1', '0.1', '96.3%', '94.0%', '11.8%', '1.93', '6.39'],
     ['T1_b0', '1', '0.0', '100.0%', '90.0%', '12.5%', '17.4', '12.93']],
    col_widths=[24, 10, 14, 20, 20, 20, 18, 18])

pdf.body_text(
    'Key observations: (1) T=8 vs T=1 difference is in RCR (2-3 vs 6-13), not accuracy. '
    '(2) beta=0 with T=8 produces KL=208.6 -- massive redundancy without compression. '
    '(3) beta=1.0 compresses to KL=0.36 with no accuracy loss, confirming rules have low intrinsic information.')

# ── 3.4 Linear probe ─────────────────────────────
pdf.add_page()
pdf.subsection_title('3.4', 'Linear Probing: Gradual Emergence of Rule Information')

pdf.add_figure(os.path.join(R, 'linear_probe.png'),
    'Linear probe accuracy at each recurrent step t=0,...,8. '
    'Red (circles): Probe P (Position, 8-class) -- maintains 97-100%. '
    'Blue (squares): Probe R (Rule, 3-class) -- rises from 33.4% (chance) to 90.5% at t=6. '
    'Dotted lines: chance levels. Rule information gradually emerges, supporting "Recurrence as Thinking Time."',
    fig_id='Figure 6', width=130)

pdf.bold_text('Table 2: Linear probe accuracy at each recurrent step.')
pdf.add_table(
    ['Step t', 'Probe P (Position)', 'Probe R (Rule)'],
    [['0 (initial)', '100.0%', '33.4%'],
     ['1', '100.0%', '55.7%'],
     ['2', '100.0%', '75.3%'],
     ['3', '100.0%', '83.9%'],
     ['4', '99.9%', '85.4%'],
     ['5', '99.5%', '88.7%'],
     ['*6', '*98.9%', '*90.5%'],
     ['7', '98.1%', '89.6%'],
     ['8 (pre-output)', '97.3%', '89.6%']],
    col_widths=[40, 55, 55])

pdf.body_text(
    'Rule information does not appear suddenly, but gradually emerges across recurrent steps. '
    'The network simultaneously maintains position memory (97%+) and rule abstraction (91%). '
    'Three phases: Early (t=0-2): local pattern extraction; Middle (t=3-5): cross-position integration; '
    'Late (t=6-8): saturation and stabilization.')

# ── 3.5 Bottleneck geometry ──────────────────────
pdf.subsection_title('3.5', 'Bottleneck Representation Geometry')

pdf.add_figure(os.path.join(R, 'bottleneck_2d.png'),
    'VIB bottleneck z_final in R^2 for Simple test sequences (n=300), colored by vertex index. '
    'Eight vertices form clearly separated clusters in approximate circular topology -- an emergent '
    'internal "coordinate system" in the 2D compressed space.',
    fig_id='Figure 7', width=105)

pdf.body_text(
    'In the extremely compressed 2D space, 8 vertex categories form clearly separated clusters with '
    'approximate circular arrangement. Adjacent vertices in the sequence are spatially proximal, '
    'reflecting that the bottleneck captures sequential adjacency relationships.')

# ── 3.6 z_final probing ─────────────────────────
pdf.subsection_title('3.6', 'Nonlinear Encoding: Direct Probing of z_final')

pdf.add_figure(os.path.join(R, 'bottleneck_position_vs_rule.png'),
    'Left: z_final colored by vertex position -- clusters visible but circularly arranged (linearly inseparable). '
    'Right: z_final colored by rule type -- all three types completely overlap. '
    'VIB compresses away rule information, retaining only position.',
    fig_id='Figure 8', width=160)

pdf.bold_text('Table 3: Linear probe accuracy on z_final (2D bottleneck).')
pdf.add_table(
    ['Probe', 'Target', 'Classes', 'Accuracy', 'Chance'],
    [['Probe P', 'Position', '8', '15.0%', '12.5%'],
     ['Probe R', 'Rule type', '3', '0.0%', '33.3%']],
    col_widths=[30, 35, 25, 30, 30])

pdf.body_text(
    'Both probes yield accuracy at or below chance. Position: 8 clusters arranged in circular topology '
    '(linearly inseparable). Rule: all three types produce overlapping z values at same positions. '
    'This reveals distributed computation: the Recurrent Transformer handles rule computation '
    '(Probe R: 33% -> 91%), the VIB bottleneck stores position (nonlinear encoding), and the output head '
    'combines both via nonlinear decoding.')

# ── 3.7 Generalization ──────────────────────────
pdf.subsection_title('3.7', 'Generalization Test: Zero-Shot on Unseen Rules')

pdf.bold_text('Table 4: Zero-shot generalization on novel rules (n=2000 each).')
pdf.add_table(
    ['Rule', 'Accuracy', 'Chance'],
    [['+3 step jumps', '16.3%', '12.5%'],
     ['-2 step jumps', '3.7%', '12.5%'],
     ['+4 steps (mirror)', '14.1%', '12.5%'],
     ['-3 step jumps', '8.6%', '12.5%']],
    col_widths=[55, 40, 40])

pdf.body_text(
    'All novel rules yield accuracy near chance (3-16%). The -2 step result (3.7%, below chance) '
    'indicates systematic directional bias. The model learned specific training rules, '
    'not abstract "modular addition on a circle."')

# ══════════════════════════════════════════════════
# 4. DISCUSSION
# ══════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('4', 'Discussion')

pdf.subsection_title('4.1', 'Information Compression as a Computational Signature of "Understanding"')
pdf.body_text(
    'The core finding is the bifurcation between beta=0 and beta>0 on the KL curve (Figure 3). '
    'Beta=0 learns rules but with KL exploding to 200+ ("rote memorization"); beta>0 achieves '
    'the same accuracy with KL=2 ("genuine understanding"). This supports: the degree of information '
    'compression (not accuracy) may distinguish memorization from understanding. This aligns with '
    '"chunking" in cognitive science: experts compress information into strategic patterns.')

pdf.subsection_title('4.2', 'Recurrent Steps as "Thinking Time"')
pdf.body_text(
    'Linear probes (Figure 6) show Probe R rising from 33% to 91% across 6 steps, while Probe P '
    'stays at 97-100%. Recurrent steps perform hierarchical information integration: early steps '
    'extract local patterns, middle steps integrate across positions, late steps stabilize. '
    'This parallels human multi-step mental simulation for logic puzzles.')

pdf.subsection_title('4.3', 'VIB Placement: Posterior Collapse and Its Resolution')
pdf.body_text(
    'Initial design: VIB inside the recurrent loop. Failure: accuracy=12.5%, KL=0 immediately. '
    'Diagnosis: Classic Posterior Collapse -- beta pressure drives mu->0, sigma->1 (pure noise); '
    'noise accumulates exponentially over T=8 steps; prediction head receives no signal. '
    'Fix: relocate VIB to readout stage (applied once after recurrence). This also better aligns '
    'with GNW theory: the "broadcast bottleneck" occurs during propagation, not thinking.')

pdf.subsection_title('4.4', 'Correspondence with Dehaene\'s Ignition Theory')

pdf.bold_text('Table 5: GNW theory <-> Computational analog mapping.')
pdf.add_table(
    ['GNW Theory', 'Computational Analog'],
    [['Conscious Ignition', 'Synchronous KL decrease + RCR convergence'],
     ['Global Broadcast', 'VIB readout (compression at broadcast)'],
     ['Workspace Capacity Limit', 'Bottleneck dim=2 (extreme compression)'],
     ['Recurrent Activation', 'Weight-sharing loops, T=8 steps'],
     ['Unconscious Processing', 'beta=0 (high KL, high RCR)'],
     ['Thinking Time', 'Probe R rises monotonically with step t']],
    col_widths=[60, 100])

pdf.subsection_title('4.5', 'Limitations and Future Work')
pdf.body_text(
    '(1) No generalization: model learns rule instances, not meta-learning. Future: MAML or richer rule sets. '
    '(2) No abrupt ignition: changes are gradual. Future: harder tasks, working memory constraints. '
    '(3) Single seed (42): robustness unverified. Future: 5-seed mean +/- std. '
    '(4) T=1 also achieves high accuracy: recurrence affects HOW rules are represented. Future: tasks requiring recurrence. '
    '(5) z_final doesn\'t linearly encode rules. Future: MLP probes, higher-dim bottlenecks.')

# ══════════════════════════════════════════════════
# 5. CONCLUSION
# ══════════════════════════════════════════════════
pdf.section_title('5', 'Conclusion')
pdf.body_text(
    '1. A minimal network learns nested rules. Single-layer weight-sharing Recurrent Transformer '
    'achieves 90-100% accuracy on Simple and Nested sequences within 100-200 steps (Random at chance).')
pdf.body_text(
    '2. Information bottleneck produces a dual effect. Beta>0 reduces KL by 85% and stabilizes RCR '
    'without accuracy loss. Beta=0 shows KL explosion to 200+ -- fundamentally different "understanding."')
pdf.body_text(
    '3. Rule information gradually emerges across recurrent steps. Probe R: 33% -> 91% over 6 steps, '
    'directly supporting "recurrence = thinking time."')
pdf.body_text(
    '4. Bottleneck exhibits emergent geometry but encodes nonlinearly. 8 separated clusters in 2D '
    'with circular topology. Rule inference relies on distributed computation (attention + decoder).')
pdf.body_text(
    '5. Architectural pitfall documented. VIB inside recurrence causes Posterior Collapse; '
    'readout-stage VIB aligns with GNW\'s broadcast mechanism.')

# ══════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════
pdf.section_title('', 'References')
pdf.set_font('Times', '', 9)
refs = [
    '[1] S. Dehaene, H. Lau, S. Kouider. "What is consciousness, and could machines have it?" Science 358(6362):486-492, 2017.',
    '[2] N. Tishby, F. C. Pereira, W. Bialek. "The information bottleneck method." arXiv physics/0004057, 2000.',
    '[3] A. A. Alemi, I. Fischer, J. V. Dillon, K. Murphy. "Deep variational information bottleneck." ICLR, 2017.',
    '[4] S. Dehaene et al. "Toward a computational theory of conscious processing." Curr. Opin. Neurobiol. 25:76-84, 2014.',
    '[5] A. Vaswani et al. "Attention is all you need." NeurIPS, 2017.',
]
for ref in refs:
    pdf.multi_cell(0, 4.5, ref)
    pdf.ln(0.5)

# ══════════════════════════════════════════════════
# APPENDIX: Individual Ignition Curves
# ══════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('A', 'Appendix: Individual Ignition Curves for All Runs')

ign_figs = [
    ('ignition_T8_b00.png',  'Figure B1: Ablation B (T=8, beta=0) -- KL explodes to 200+, RCR rises continuously.'),
    ('ignition_T8_b001.png', 'Figure B2: T=8, beta=0.01 -- Weak bottleneck. KL rises then falls.'),
    ('ignition_T8_b01.png',  'Figure B3: T=8, beta=0.1 (sweep run, 1500 steps).'),
    ('ignition_T8_b10.png',  'Figure B4: T=8, beta=1.0 -- Strongest compression (KL -> 0.36).'),
    ('ignition_T1_b01.png',  'Figure B5: Ablation A (T=1, beta=0.1) -- High RCR, no convergence.'),
    ('ignition_T1_b0.png',   'Figure B6: Ablation C (T=1, beta=0) -- Highest RCR (~13).'),
]

for fname, caption in ign_figs:
    fpath = os.path.join(R, fname)
    if os.path.exists(fpath):
        pdf.add_figure(fpath, caption, width=130)

# ── Appendix B: Training snapshots ────────────────
pdf.add_page()
pdf.section_title('B', 'Appendix: Complete Training Dynamics (Main Model)')

pdf.bold_text('Table A1: Selected training snapshots for main_b01 (T=8, beta=0.1, 3000 steps).')
pdf.add_table(
    ['Step', 'beta', 'Simple', 'Nested', 'Random', 'KL', 'RCR', 'Time(s)'],
    [['100', '0.007', '92.6%', '70.4%', '12.6%', '13.72', '3.09', '3.3'],
     ['300', '0.020', '95.2%', '95.3%', '12.3%', '8.59', '2.83', '9.7'],
     ['500', '0.032', '92.5%', '97.6%', '12.5%', '5.90', '2.69', '16.1'],
     ['1000', '0.058', '97.6%', '92.9%', '12.5%', '2.68', '2.50', '32.8'],
     ['1500', '0.076', '98.8%', '91.3%', '12.5%', '2.18', '2.42', '49.9'],
     ['2000', '0.087', '96.2%', '93.9%', '12.2%', '2.17', '2.41', '67.2'],
     ['2500', '0.093', '94.8%', '94.8%', '12.4%', '2.09', '2.33', '84.6'],
     ['3000', '0.096', '93.8%', '96.3%', '12.6%', '2.01', '2.38', '101.8']],
    col_widths=[18, 16, 22, 22, 22, 18, 18, 22])

pdf.body_text('Total training time: 101.8 seconds on CPU (Intel, Windows 11).')

# ── Save ──────────────────────────────────────────
output_path = os.path.join(BASE, 'LAB_REPORT.pdf')
pdf.output(output_path)
print(f'[OK] PDF report generated: {output_path}')
print(f'     File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')
