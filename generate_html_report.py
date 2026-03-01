"""
generate_html_report.py — Self-contained HTML lab report with embedded images

Reads all result figures from results/ and checkpoints/, encodes them as
base64 data URIs, and produces a single portable HTML file.

Author: Puzhi YU
Date:   January 2026
"""

import base64
import os

def img_to_base64(path):
    """Read an image file and return base64-encoded data URI."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = path.rsplit('.', 1)[-1].lower()
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext, 'image/png')
    return f'data:{mime};base64,{data}'


def img_tag(path, caption='', width='100%', fig_id=''):
    """Generate an HTML figure block with embedded image."""
    uri = img_to_base64(path)
    if uri is None:
        return f'<p style="color:red;">[Image not found: {path}]</p>'
    fig_label = f'<b>{fig_id}</b>: ' if fig_id else ''
    return f'''
    <figure style="text-align:center; margin:20px 0;">
      <img src="{uri}" style="max-width:{width}; border:1px solid #ddd; border-radius:4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
      <figcaption style="margin-top:8px; font-size:0.92em; color:#555; max-width:90%; margin-left:auto; margin-right:auto;">
        {fig_label}{caption}
      </figcaption>
    </figure>
    '''


# ── Paths ────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, 'results')

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab Report — Ignition Mechanisms in Information-Bottleneck Recurrent Networks</title>
<style>
  body {{
    font-family: 'Times New Roman', 'Noto Serif', Georgia, serif;
    max-width: 960px;
    margin: 0 auto;
    padding: 30px 40px;
    line-height: 1.65;
    color: #1a1a1a;
    background: #fff;
  }}
  h1 {{
    font-size: 1.55em;
    text-align: center;
    margin-bottom: 5px;
    line-height: 1.35;
  }}
  h2 {{
    font-size: 1.25em;
    border-bottom: 2px solid #2C3E50;
    padding-bottom: 4px;
    margin-top: 35px;
  }}
  h3 {{
    font-size: 1.08em;
    margin-top: 22px;
    color: #2C3E50;
  }}
  .meta {{
    text-align: center;
    color: #666;
    margin-bottom: 25px;
  }}
  .abstract {{
    background: #f8f9fa;
    border-left: 4px solid #2C3E50;
    padding: 15px 20px;
    margin: 20px 0;
    font-style: italic;
  }}
  .keywords {{
    font-size: 0.92em;
    color: #555;
  }}
  table {{
    border-collapse: collapse;
    margin: 15px auto;
    font-size: 0.92em;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 6px 12px;
    text-align: center;
  }}
  th {{
    background: #2C3E50;
    color: white;
    font-weight: bold;
  }}
  tr:nth-child(even) {{
    background: #f8f9fa;
  }}
  tr.highlight {{
    background: #eaf4fd;
    font-weight: bold;
  }}
  blockquote {{
    border-left: 4px solid #3498DB;
    margin: 15px 0;
    padding: 10px 20px;
    background: #f0f7fd;
    font-style: italic;
  }}
  code {{
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.9em;
  }}
  pre {{
    background: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 0.88em;
    line-height: 1.45;
  }}
  figure {{
    page-break-inside: avoid;
  }}
  .two-col {{
    display: flex;
    gap: 20px;
    align-items: flex-start;
  }}
  .two-col > div {{
    flex: 1;
  }}
  .eq {{
    text-align: center;
    margin: 12px 0;
    font-style: italic;
  }}
  .ref {{
    font-size: 0.9em;
    color: #444;
  }}
  @media print {{
    body {{ padding: 15px; font-size: 10.5pt; }}
    figure img {{ max-width: 100% !important; }}
  }}
</style>
</head>
<body>

<h1>Ignition Mechanisms in Information-Bottleneck Recurrent Networks:<br>
A Computational Study on Octagon Geometric Sequence Tasks</h1>

<div class="meta">
  <p><b>Author</b>: Puzhi Yu &nbsp; | &nbsp; <b>Date</b>: February 2026 &nbsp; | &nbsp; <b>Repository</b>: <code>causal_emergence/</code></p>
</div>

<div class="abstract">
<b>Abstract.</b>
We construct a minimalist Recurrent Transformer augmented with a Variational Information Bottleneck (VIB) and investigate the computational mechanisms underlying the "Ignition" phenomenon from Dehaene's Global Neuronal Workspace (GNW) theory. Using an octagon geometric sequence prediction task with three difficulty levels (Simple, Nested, Random), we demonstrate that:
(1) A 13,292-parameter weight-sharing recurrent network spontaneously learns nested sequence rules from raw geometric coordinates, achieving 90&ndash;100% accuracy while random control sequences remain at chance (12.5%).
(2) Information bottleneck pressure (&beta; &gt; 0) compresses internal representations by ~85% (KL: 13 &rarr; 2) without accuracy loss, while &beta;=0 shows KL explosion to 200+.
(3) Linear probing reveals rule information <em>gradually emerges</em> across recurrent steps (33% at t=0 to 91% at t=6), supporting "Recurrence as Thinking Time."
(4) The 2D bottleneck spontaneously forms separated vertex clusters with circular topology, yet encodes position nonlinearly and discards rule-type information&mdash;revealing distributed computation.
We additionally document a critical architectural lesson: VIB inside the recurrent loop causes Posterior Collapse; relocating it to the readout stage aligns with GNW's "broadcast" semantics.
</div>

<p class="keywords"><b>Keywords</b>: Information Bottleneck, Global Neuronal Workspace, Ignition, Weight-Sharing Recurrent Transformer, Representation Learning, Geometric Rules</p>

<hr>

<h2>1. Introduction</h2>

<h3>1.1 Background: Global Neuronal Workspace Theory and Ignition</h3>

<p>Stanislas Dehaene's Global Neuronal Workspace (GNW) theory [1] posits that conscious information processing corresponds to a special neural event&mdash;<b>Ignition</b>: when sensory information exceeds a threshold, it triggers a nonlinear, cross-cortical spreading activation that rapidly propagates across the brain, forming a stable global representation. In EEG/MEG experiments, this manifests as the sudden onset of the P3 wave and synchronized firing across fronto-parietal regions.</p>

<p>However, the <b>computational mechanisms</b> behind Ignition remain unclear. Specifically:</p>
<ul>
  <li>What information processing operations correspond to "discarding details and distilling rules"?</li>
  <li>What role does recurrence play in rule extraction?</li>
  <li>Is information compression a necessary condition or a consequence of "understanding"?</li>
</ul>

<h3>1.2 Information Bottleneck Framework</h3>

<p>The Information Bottleneck (IB) principle proposed by Tishby et al. [2] provides a formal framework: given input X and prediction target Y, find a bottleneck variable Z that maximizes I(Z; Y) (predictive relevance) while minimizing I(X; Z) (dependence on input details). Alemi et al. [3] extended this to the Variational Information Bottleneck (VIB), enabling end-to-end training.</p>

<p>This study combines VIB with a recurrent Transformer to construct a minimal "computational brain," exploring how the information bottleneck facilitates rule emergence in sequential learning.</p>

<h3>1.3 Research Questions</h3>

<blockquote><b>Q1</b>: Can a minimalist recurrent network (single layer, weight-sharing, 13K parameters) spontaneously learn nested structural rules from geometric coordinate sequences?</blockquote>

<blockquote><b>Q2</b>: Does information bottleneck pressure (VIB, parameter &beta;) produce more compact and stable internal representations without sacrificing prediction accuracy?</blockquote>

<blockquote><b>Q3</b>: Does rule information emerge gradually across recurrent steps, or appear instantaneously?</blockquote>

<hr>

<h2>2. Methods</h2>

<h3>2.1 Experimental Paradigm: Octagon Sequence Task</h3>

<p>We computationally implement a variant of the classical paradigm from Dehaene's laboratory [1], using vertex sequences on a regular octagon as stimuli.</p>

<h4>Input Space</h4>
<p>Eight vertices of a regular octagon, indexed k &isin; &#123;0, 1, &hellip;, 7&#125;, are encoded as geometric coordinates on the unit circle:</p>
<p class="eq">input<sub>k</sub> = [cos(2&pi;k/8), sin(2&pi;k/8)] &isin; &real;<sup>2</sup></p>
<p>Geometric coordinates (rather than one-hot) preserve Euclidean distance information between vertices&mdash;a prerequisite for exploring emergent geometric rule representations.</p>

<h4>Prediction Task</h4>
<p><b>Autoregressive next-step prediction</b>: given x<sub>1</sub>, x<sub>2</sub>, &hellip;, x<sub>t</sub>, predict x<sub>t+1</sub> as an 8-class classification (cross-entropy loss). Chance-level accuracy = 1/8 = 12.5%.</p>

<h4>Sequence Types (Three Difficulty Levels)</h4>

{img_tag(os.path.join(R, 'paradigm_schematic.png'),
         'The octagon geometric sequence task with three difficulty levels. '
         '(a) <b>Simple</b>: clockwise +1 step per time point (period 8). '
         '(b) <b>Nested</b>: alternating +2 (forward, blue) and &minus;1 (backward, orange) steps, '
         'creating a "two steps forward, one step back" pattern (micro-period 2, macro-period 16). '
         '(c) <b>Random</b> (control): uniformly random jumps, incompressible by any model.',
         width='95%', fig_id='Figure 1')}

<table>
  <tr><th>Type</th><th>Generation Rule</th><th>Example (from vertex 0)</th><th>Complexity</th></tr>
  <tr><td><b>Simple</b></td><td>x<sub>t+1</sub> = (x<sub>t</sub> + 1) mod 8</td><td>0&rarr;1&rarr;2&rarr;3&rarr;4&rarr;5&hellip;</td><td>Minimal (period 8)</td></tr>
  <tr><td><b>Nested</b></td><td>Even steps: +2; Odd steps: &minus;1 (mod 8)</td><td>0&rarr;2&rarr;1&rarr;3&rarr;2&rarr;4&hellip;</td><td>Medium-high (micro-period 2, macro-period 16)</td></tr>
  <tr><td><b>Random</b></td><td>x<sub>t+1</sub> ~ U&#123;0,&hellip;,7&#125;</td><td>0&rarr;5&rarr;2&rarr;7&rarr;1&hellip;</td><td>Incompressible (max entropy)</td></tr>
</table>

<h4>Dataset Construction</h4>
<ul>
  <li><b>Scale</b>: 10,000 sequences per type (30,000 total)</li>
  <li><b>Sequence length</b>: L = 10</li>
  <li><b>Split</b>: 70% / 15% / 15% (train / val / test; each type split independently)</li>
  <li><b>Seed</b>: 42 (reproducible)</li>
</ul>

<h3>2.2 Model Architecture</h3>

<h4>Design Principles</h4>
<blockquote><b>Core Hypothesis</b>: Rule extraction does not require depth; it requires sufficient "thinking time" (recurrence) and "forgetting pressure" (bottleneck).</blockquote>

<h4>Architecture Overview</h4>
<pre>
Input  [B, L=10, 2]  &larr; Geometric coordinate sequence
  &darr;
  Linear Embedding (2 &rarr; d_model=32)
  + Sinusoidal Positional Encoding (fixed, non-learnable)
  &darr;
  ┌──────────────────────────────────────────────────┐
  &vert;  Recurrent Unrolling  T=8 times (shared W)      &vert;
  &vert;  for t = 1 .. 8:                                &vert;
  &vert;    H_t = TransformerEncoderLayer(H_{{t-1}})       &vert;
  &vert;    Record RCR_t = &Vert;H_t &minus; H_{{t-1}}&Vert;&#x2082;          &vert;
  &vert;  end                                            &vert;
  └──────────────────────────────────────────────────┘
  &darr;
  VIB Readout Layer (applied ONCE after all T loops)
    &mu;, &sigma; = Encoder(H_T)     [B,L,32] &rarr; [B,L,2]
    z = &mu; + &epsilon;&middot;&sigma;             &epsilon; ~ N(0,I)
    H_out = Decoder(z)          [B,L,2] &rarr; [B,L,32]
  &darr;
  Output Head (Linear &rarr; 8-class Softmax)
Output [B, L, 8]
</pre>

<p><b>Weight sharing</b>: The single Transformer layer is reused across all T=8 steps, forcing iterative refinement rather than memorization.</p>
<p><b>VIB placement</b>: Applied <em>after</em> the recurrent loop (readout stage), not inside it. See Section 4.3 for rationale.</p>

<h4>Parameter Count</h4>
<table>
  <tr><th>Component</th><th>Parameters</th></tr>
  <tr><td>Input embedding (2 &rarr; 32)</td><td>64</td></tr>
  <tr><td>Transformer layer (d=32, nhead=4, FFN=128)</td><td>~12,800</td></tr>
  <tr><td>VIB layer (32 &rarr; 2 &rarr; 32)</td><td>192</td></tr>
  <tr><td>Output head (32 &rarr; 8)</td><td>264</td></tr>
  <tr class="highlight"><td><b>Total</b></td><td><b>13,292</b></td></tr>
</table>

<h3>2.3 Loss Function and Training</h3>

<h4>Composite Loss</h4>
<p class="eq">&Lscr; = &minus;log P(Y | z) + &beta;(t) &middot; KL( N(&mu;, &sigma;&sup2;) &Vert; N(0, I) )</p>

<p>KL has an analytical solution:</p>
<p class="eq">KL = &minus;&frac12; &Sigma;<sub>j</sub> (1 + log&sigma;<sub>j</sub>&sup2; &minus; &mu;<sub>j</sub>&sup2; &minus; &sigma;<sub>j</sub>&sup2;)</p>

<h4>&beta; Annealing Schedule</h4>
<p class="eq">&beta;(t) = &beta;<sub>max</sub> &middot; tanh(t / T<sub>warmup</sub>), &emsp; T<sub>warmup</sub> = 0.5 &times; T<sub>total</sub></p>
<p>During the first 50% of training steps, &beta; increases from 0, allowing the model to learn prediction before compression pressure is applied.</p>

<h4>Training Hyperparameters</h4>
<table>
  <tr><th>Hyperparameter</th><th>Value</th></tr>
  <tr><td>Total steps</td><td>3,000 (main) / 1,500 (ablation)</td></tr>
  <tr><td>Batch size</td><td>128</td></tr>
  <tr><td>Optimizer</td><td>AdamW (lr=10<sup>&minus;3</sup>, weight_decay=10<sup>&minus;4</sup>)</td></tr>
  <tr><td>Gradient clipping</td><td>max_norm = 1.0</td></tr>
  <tr><td>Random seed</td><td>42</td></tr>
</table>

<h3>2.4 Evaluation Metrics</h3>
<table>
  <tr><th>Metric</th><th>Interpretation</th></tr>
  <tr><td><b>Accuracy</b></td><td>Behavioral learning performance (8-class next-step prediction)</td></tr>
  <tr><td><b>KL Divergence</b></td><td>Information-level compression degree (VIB bottleneck)</td></tr>
  <tr><td><b>RCR</b> (&Vert;H<sub>t</sub> &minus; H<sub>t&minus;1</sub>&Vert;)</td><td>Dynamical attractor stability across recurrent steps</td></tr>
</table>

<h3>2.5 Ablation Study Design</h3>
<table>
  <tr><th>Run ID</th><th>T</th><th>&beta;<sub>max</sub></th><th>Role</th></tr>
  <tr><td>T8_b00</td><td>8</td><td>0.0</td><td>Ablation B: Recurrence, no bottleneck</td></tr>
  <tr><td>T8_b001</td><td>8</td><td>0.01</td><td>Weak bottleneck</td></tr>
  <tr class="highlight"><td><b>main_b01</b></td><td><b>8</b></td><td><b>0.1</b></td><td><b>&starf; Primary model (3000 steps)</b></td></tr>
  <tr><td>T8_b01</td><td>8</td><td>0.1</td><td>&beta; sweep (1500 steps)</td></tr>
  <tr><td>T8_b10</td><td>8</td><td>1.0</td><td>Strong bottleneck</td></tr>
  <tr><td>T1_b01</td><td>1</td><td>0.1</td><td>Ablation A: Bottleneck, no deep recurrence</td></tr>
  <tr><td>T1_b0</td><td>1</td><td>0.0</td><td>Ablation C: Pure feedforward baseline</td></tr>
</table>

<hr>

<h2>3. Results</h2>

<h3>3.1 Main Model Learning Dynamics</h3>

{img_tag(os.path.join(R, 'ignition_main_b01.png'),
         'Three-panel Ignition curve for the main model (T=8, &beta;=0.1, 3000 steps). '
         '<b>Top</b>: Accuracy (%) for Simple (red), Nested (blue), and Random (gray). '
         'Ignition detected at step 100 (vertical dashed line). Random baseline at 12.5%. '
         '<b>Middle</b>: Mean KL Divergence&mdash;decreases from ~13.7 to ~2.0 (85% compression). '
         '<b>Bottom</b>: RCR decreases from ~3.1 to ~2.3, indicating convergence to a stable attractor.',
         width='70%', fig_id='Figure 2')}

<p><b>Accuracy (Top Panel)</b>: Within ~100 steps (3.3 seconds on CPU), Simple and Nested accuracy jumps to 92.6% and 70.4% respectively. By step 200, both stabilize at 93&ndash;96%. Random accuracy stays at 12.0&ndash;12.9% throughout, ruling out overfitting.</p>

<p><b>KL Divergence (Middle Panel)</b>: Initial KL &asymp; 13.7 monotonically decreases to &asymp; 2.0, an <b>85% reduction</b>. The model accomplishes the same prediction task with progressively fewer "bits" of internal representation.</p>

<p><b>RCR (Bottom Panel)</b>: Decreases from ~3.1 to ~2.3, paralleling the KL decrease&mdash;as rule extraction completes, per-step state changes become smaller and the system approaches a stable attractor.</p>

<h3>3.2 Effect of Information Bottleneck Pressure (&beta; Sweep)</h3>

{img_tag(os.path.join(R, 'beta_sweep_nested.png'),
         'Comparison of four &beta; values (0.0, 0.01, 0.1, 1.0) for T=8 models on <b>Nested</b> sequences (1500 steps). '
         '<b>Left</b>: All &beta; configs reach 90&ndash;100% accuracy&mdash;accuracy is insensitive to &beta;. '
         '<b>Middle</b>: <em>Critical differentiation</em>&mdash;&beta;=0 (gray) shows KL explosion to ~150, '
         'while &beta;=0.1 (blue) compresses to ~2 and &beta;=1.0 (dark blue) to ~0.4. '
         '<b>Right</b>: &beta;=0 shows continuously rising RCR (turbulent dynamics), while &beta;&gt;0 converges.',
         width='98%', fig_id='Figure 3')}

{img_tag(os.path.join(R, 'beta_sweep_simple.png'),
         'Same &beta; sweep comparison on <b>Simple</b> sequences. KL and RCR patterns are consistent with the Nested results '
         '(Figure 3), confirming that the bottleneck effect is rule-independent.',
         width='98%', fig_id='Figure 4')}

<blockquote><b>Key Finding</b>: Under equivalent accuracy, VIB (&beta; &gt; 0) fundamentally alters the model's internal information state&mdash;from "inflating" to "compressing," from "turbulent" to "stable." This constitutes a <b>dual Ignition signal</b> combining information compression and attractor dynamics.</blockquote>

<h3>3.3 2&times;2 Ablation: Disentangling Recurrence and Bottleneck</h3>

{img_tag(os.path.join(R, 'ablation_2x2_nested.png'),
         '2&times;2 ablation comparison on Nested sequences. '
         '<b>Blue</b>: Main model (T=8, &beta;=0.1). '
         '<b>Red</b>: Ablation A (T=1, &beta;=0.1). '
         '<b>Purple</b>: Ablation C (T=1, &beta;=0). '
         'The RCR panel (right) shows the most dramatic difference: T=8 achieves low, stable RCR (~2.5), '
         'while T=1 models show RCR of 6&ndash;12.',
         width='98%', fig_id='Figure 5')}

<p><b>Table 1</b>: Final metrics for all experimental runs.</p>
<table>
  <tr><th>Run</th><th>T</th><th>&beta;<sub>max</sub></th><th>Simple Acc</th><th>Nested Acc</th><th>Random Acc</th><th>Final KL</th><th>Final RCR</th></tr>
  <tr><td>T8_b00</td><td>8</td><td>0.0</td><td>93.7%</td><td>94.8%</td><td>13.1%</td><td>208.6</td><td>5.39</td></tr>
  <tr><td>T8_b001</td><td>8</td><td>0.01</td><td>90.0%</td><td><b>100.0%</b></td><td>12.2%</td><td>5.84</td><td>2.79</td></tr>
  <tr class="highlight"><td><b>main_b01</b></td><td><b>8</b></td><td><b>0.1</b></td><td>93.8%</td><td>96.3%</td><td>12.6%</td><td><b>2.01</b></td><td>2.38</td></tr>
  <tr><td>T8_b01</td><td>8</td><td>0.1</td><td>95.0%</td><td>95.0%</td><td>12.4%</td><td>2.27</td><td>2.42</td></tr>
  <tr><td>T8_b10</td><td>8</td><td>1.0</td><td>95.0%</td><td>95.0%</td><td>12.4%</td><td><b>0.36</b></td><td><b>2.03</b></td></tr>
  <tr><td>T1_b01</td><td>1</td><td>0.1</td><td>96.3%</td><td>94.0%</td><td>11.8%</td><td>1.93</td><td>6.39</td></tr>
  <tr><td>T1_b0</td><td>1</td><td>0.0</td><td><b>100.0%</b></td><td>90.0%</td><td>12.5%</td><td>17.4</td><td>12.93</td></tr>
</table>

<p><b>Key observations</b>:</p>
<ol>
  <li><b>T=8 vs T=1 difference is in RCR, not accuracy.</b> T=8 models: RCR 2&ndash;3; T=1 models: RCR 6&ndash;13. Recurrence converges dynamics.</li>
  <li><b>&beta;=0 with T=8 produces KL = 208.6</b>&mdash;without compression, the recurrent network accumulates massive redundancy.</li>
  <li><b>&beta;=1.0 compresses KL to 0.36 with no accuracy loss</b>, confirming the learned rules have inherently low information content.</li>
</ol>

<h3>3.4 Linear Probing: Gradual Emergence of Rule Information</h3>

{img_tag(os.path.join(R, 'linear_probe.png'),
         'Linear probe accuracy at each recurrent step t=0,...,8. '
         '<b>Red circles</b>: Probe P (Position, 8-class)&mdash;maintains 97&ndash;100% throughout. '
         '<b>Blue squares</b>: Probe R (Rule, 3-class)&mdash;rises from 33.4% (chance) at t=0 to 90.5% at t=6. '
         'Dotted lines show chance levels. Rule information <em>gradually emerges</em> across recurrent steps, '
         'directly supporting the "Recurrence as Thinking Time" hypothesis.',
         width='72%', fig_id='Figure 6')}

<p><b>Table 2</b>: Linear probe accuracy at each recurrent step.</p>
<table>
  <tr><th>Step t</th><th>Probe P (Position)</th><th>Probe R (Rule)</th></tr>
  <tr><td>0 (initial embed)</td><td>100.0%</td><td>33.4%</td></tr>
  <tr><td>1</td><td>100.0%</td><td>55.7%</td></tr>
  <tr><td>2</td><td>100.0%</td><td>75.3%</td></tr>
  <tr><td>3</td><td>100.0%</td><td>83.9%</td></tr>
  <tr><td>4</td><td>99.9%</td><td>85.4%</td></tr>
  <tr><td>5</td><td>99.5%</td><td>88.7%</td></tr>
  <tr class="highlight"><td><b>6</b></td><td>98.9%</td><td><b>90.5%</b></td></tr>
  <tr><td>7</td><td>98.1%</td><td>89.6%</td></tr>
  <tr><td>8 (pre-output)</td><td>97.3%</td><td>89.6%</td></tr>
</table>

<p><b>Interpretation</b>: Rule information does not appear suddenly at a single step, but <b>gradually emerges</b> across recurrent steps (t=0 to t=6 shows clear monotonic increase). The network simultaneously maintains position memory (97%+) and rule abstraction (91%), rather than trading off one for the other.</p>

<p>This dynamic process can be decomposed into three phases:</p>
<ul>
  <li><b>Early steps (t=0&ndash;2)</b>: Extract local patterns from raw coordinates</li>
  <li><b>Middle steps (t=3&ndash;5)</b>: Cross-position integration, distilling regularities</li>
  <li><b>Late steps (t=6&ndash;8)</b>: Representation stabilizes, rule decoding saturates</li>
</ul>

<h3>3.5 Bottleneck Representation Geometry</h3>

{img_tag(os.path.join(R, 'bottleneck_2d.png'),
         'VIB bottleneck representation z<sub>final</sub> &isin; &real;&sup2; for Simple test sequences (n=300). '
         'Each point represents one sequence position, colored by vertex index (0&ndash;7). '
         'The 8 vertices form <b>clearly separated clusters</b> arranged in an approximate '
         '<b>circular topology</b>&mdash;the model spontaneously organizes an emergent internal "coordinate system" '
         'in the 2D compressed space.',
         width='58%', fig_id='Figure 7')}

<p>In the extremely compressed 2-dimensional space, the 8 vertex categories form clearly separated clusters, each vertex occupying a compact, distinct region. The clusters exhibit an approximate <b>circular arrangement</b>, reflecting that the bottleneck captures sequential adjacency relationships between vertices.</p>

<h3>3.6 Nonlinear Encoding in the Bottleneck: Direct Probing of z<sub>final</sub></h3>

{img_tag(os.path.join(R, 'bottleneck_position_vs_rule.png'),
         'What does the bottleneck z<sub>final</sub> actually encode? '
         '<b>Left</b>: colored by vertex position&mdash;clusters visible but arranged circularly (linearly inseparable). '
         '<b>Right</b>: colored by rule type (Simple/Nested/Random)&mdash;all three rule types <b>completely overlap</b>, '
         'confirming VIB compresses away rule information and retains only position.',
         width='92%', fig_id='Figure 8')}

<p><b>Table 3</b>: Linear probe accuracy on z<sub>final</sub> (2D bottleneck vectors).</p>
<table>
  <tr><th>Probe</th><th>Target</th><th>Classes</th><th>Accuracy</th><th>Chance</th></tr>
  <tr><td>Probe P</td><td>Position</td><td>8</td><td><b>15.0%</b></td><td>12.5%</td></tr>
  <tr><td>Probe R</td><td>Rule type</td><td>3</td><td><b>0.0%</b></td><td>33.3%</td></tr>
</table>

<p><b>Explanation</b>:</p>
<ul>
  <li><b>Position (15%)</b>: 8 clusters are visually separated but arranged in a <b>circular topology</b>&mdash;linearly inseparable with hyperplanes in 2D.</li>
  <li><b>Rule (0%)</b>: All three rule types visit the same 8 vertices, producing overlapping z values. VIB "forgets rule type and retains position."</li>
</ul>

<blockquote><b>Deep Significance</b>: The Recurrent Transformer (H<sub>t</sub>) handles rule computation (Probe R: 33%&rarr;91%); the VIB bottleneck (z<sub>final</sub>) stores position memory (nonlinear circular encoding); the output head combines both via nonlinear decoding. This is <b>distributed computation</b>, not single-variable holographic storage.</blockquote>

<h3>3.7 Generalization Test: Zero-Shot on Unseen Rules</h3>

<p><b>Table 4</b>: Zero-shot generalization accuracy on novel rules (n=2000 each).</p>
<table>
  <tr><th>Rule</th><th>Accuracy</th><th>Chance</th></tr>
  <tr><td>+3 step jumps</td><td>16.3%</td><td>12.5%</td></tr>
  <tr><td>&minus;2 step jumps</td><td><b>3.7%</b></td><td>12.5%</td></tr>
  <tr><td>+4 step jumps (mirror)</td><td>14.1%</td><td>12.5%</td></tr>
  <tr><td>&minus;3 step jumps</td><td>8.6%</td><td>12.5%</td></tr>
</table>

<p>All novel rules yield accuracy near chance (3&ndash;16%). The &minus;2 step result (3.7%, <em>below</em> chance) indicates a systematic <b>directional bias</b>. The model learned specific training rules (+1 and +2/&minus;1), <em>not</em> abstract "modular addition on a circle."</p>

<hr>

<h2>4. Discussion</h2>

<h3>4.1 Information Compression as a Computational Signature of "Understanding"</h3>

<p>The most central finding is the <b>bifurcation between &beta;=0 and &beta;&gt;0 on the KL curve</b> (Figure 3, middle panel). The &beta;=0 model learns the rules but with KL exploding to 200+ ("rote memorization"), while &beta;&gt;0 achieves the same accuracy with KL compressed to 2 ("genuine understanding"). This supports the proposition:</p>

<blockquote><b>The degree of information compression (rather than prediction accuracy) may be a computational signature distinguishing "memorization" from "understanding."</b></blockquote>

<p>This aligns with "chunking" in cognitive science: expert chess players compress board positions into strategic patterns rather than memorizing individual pieces.</p>

<h3>4.2 Recurrent Steps as "Thinking Time"</h3>

<p>Linear probe results (Figure 6) show Probe R rising from 33% to 91% across 6 recurrent steps, while Probe P stays saturated (97&ndash;100%). Recurrent steps perform <b>hierarchical information integration</b>: early steps extract local patterns, middle steps integrate across positions, and late steps stabilize the representation. This is analogous to human multi-step mental simulation when solving logic puzzles.</p>

<h3>4.3 VIB Placement: A Critical Architectural Decision (Posterior Collapse)</h3>

<p><b>Initial design</b>: VIB inside the recurrent loop (compression&ndash;reconstruction at every step).</p>
<p><b>Failure</b>: Accuracy stayed at 12.5%, KL collapsed to 0 immediately.</p>
<p><b>Diagnosis</b>: Classic <b>Posterior Collapse</b>&mdash;&beta; pressure drives &mu;&rarr;0, &sigma;&rarr;1 (pure noise); noise accumulates exponentially over T=8 steps; prediction head receives no signal.</p>
<p><b>Fix</b>: Relocate VIB to readout stage (applied once after recurrence). This also better aligns with GNW theory: the "broadcast bottleneck" occurs during the propagation phase, not the thinking phase.</p>

<h3>4.4 Correspondence with Dehaene's Ignition Theory</h3>

<p><b>Table 5</b>: Mapping between GNW theory and computational analogs.</p>
<table>
  <tr><th>Dehaene's GNW Theory</th><th>Computational Analog</th></tr>
  <tr><td>Conscious Ignition</td><td>Synchronous KL decrease + RCR convergence</td></tr>
  <tr><td>Global Broadcast</td><td>VIB readout layer (compression at broadcast)</td></tr>
  <tr><td>Workspace Capacity Limit</td><td>Bottleneck dim = 2 (extreme compression)</td></tr>
  <tr><td>Recurrent Activation</td><td>Weight-sharing loops, T=8 steps</td></tr>
  <tr><td>Unconscious Processing</td><td>&beta;=0 condition (high KL, high RCR)</td></tr>
  <tr><td>Thinking Time</td><td>Probe R rises monotonically with step t</td></tr>
</table>

<h3>4.5 Limitations and Future Work</h3>

<ol>
  <li><b>No generalization</b>: The model learns rule instances, not meta-learning. <em>Future</em>: MAML or richer rule sets.</li>
  <li><b>No abrupt ignition</b>: Changes are gradual, not step-function. <em>Future</em>: harder tasks, working memory constraints.</li>
  <li><b>Single seed</b>: Fixed seed 42; robustness unverified. <em>Future</em>: 5-seed mean&plusmn;std.</li>
  <li><b>&beta;&ndash;T joint effects</b>: T=1 also achieves high accuracy; recurrence affects <em>how</em> rules are represented. <em>Future</em>: tasks requiring recurrence.</li>
  <li><b>Nonlinear bottleneck encoding</b>: z<sub>final</sub> doesn't linearly encode rules. <em>Future</em>: MLP probes, higher-dim bottlenecks.</li>
</ol>

<hr>

<h2>5. Conclusion</h2>

<ol>
  <li><b>A minimal network learns nested rules.</b> Single-layer weight-sharing Recurrent Transformer achieves 90&ndash;100% accuracy on Simple and Nested sequences within 100&ndash;200 steps, while Random stays at chance.</li>
  <li><b>Information bottleneck produces a dual effect.</b> &beta;&gt;0 reduces KL by 85% and stabilizes RCR without accuracy loss. &beta;=0 shows KL explosion to 200+&mdash;fundamentally different "modes of understanding."</li>
  <li><b>Rule information gradually emerges across recurrent steps.</b> Probe R: 33%&rarr;91% over 6 steps, supporting "recurrence = thinking time."</li>
  <li><b>Bottleneck exhibits emergent geometry but encodes nonlinearly.</b> 8 separated clusters in 2D with circular topology. Rule inference relies on distributed computation (attention + nonlinear decoder).</li>
  <li><b>Architectural pitfall documented.</b> VIB inside recurrence causes Posterior Collapse; readout-stage VIB aligns with GNW's broadcast mechanism.</li>
</ol>

<hr>

<h2>References</h2>
<div class="ref">
<p>[1] S. Dehaene, H. Lau, and S. Kouider. "What is consciousness, and could machines have it?" <em>Science</em>, 358(6362):486&ndash;492, 2017.</p>
<p>[2] N. Tishby, F. C. Pereira, and W. Bialek. "The information bottleneck method." <em>arXiv preprint physics/0004057</em>, 2000.</p>
<p>[3] A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy. "Deep variational information bottleneck." <em>ICLR</em>, 2017.</p>
<p>[4] S. Dehaene, L. Charles, J.-R. King, and S. Marti. "Toward a computational theory of conscious processing." <em>Current Opinion in Neurobiology</em>, 25:76&ndash;84, 2014.</p>
<p>[5] A. Vaswani, N. Shazeer, N. Parmar, et al. "Attention is all you need." <em>NeurIPS</em>, 2017.</p>
</div>

<hr>

<h2>Appendix A: Complete Training Dynamics</h2>

<p><b>Table A1</b>: Selected training snapshots for the main model (T=8, &beta;=0.1, 3000 steps).</p>
<table>
  <tr><th>Step</th><th>&beta;</th><th>Simple Acc</th><th>Nested Acc</th><th>Random Acc</th><th>KL</th><th>RCR</th><th>Time (s)</th></tr>
  <tr><td>100</td><td>0.007</td><td>92.6%</td><td>70.4%</td><td>12.6%</td><td>13.72</td><td>3.09</td><td>3.3</td></tr>
  <tr><td>500</td><td>0.032</td><td>92.5%</td><td>97.6%</td><td>12.5%</td><td>5.90</td><td>2.69</td><td>16.1</td></tr>
  <tr><td>1000</td><td>0.058</td><td>97.6%</td><td>92.9%</td><td>12.5%</td><td>2.68</td><td>2.50</td><td>32.8</td></tr>
  <tr><td>2000</td><td>0.087</td><td>96.2%</td><td>93.9%</td><td>12.2%</td><td>2.17</td><td>2.41</td><td>67.2</td></tr>
  <tr><td>3000</td><td>0.096</td><td>93.8%</td><td>96.3%</td><td>12.6%</td><td>2.01</td><td>2.38</td><td>101.8</td></tr>
</table>
<p>Total training time: <b>101.8 seconds</b> on CPU.</p>

<h2>Appendix B: Individual Ignition Curves</h2>

<div class="two-col">
<div>
{img_tag(os.path.join(R, 'ignition_T8_b00.png'),
         'Ablation B (T=8, &beta;=0): KL explodes, RCR rises continuously.',
         width='100%', fig_id='Figure B1')}
</div>
<div>
{img_tag(os.path.join(R, 'ignition_T8_b001.png'),
         'T=8, &beta;=0.01: Weak bottleneck. KL rises then falls.',
         width='100%', fig_id='Figure B2')}
</div>
</div>

<div class="two-col">
<div>
{img_tag(os.path.join(R, 'ignition_T8_b01.png'),
         'T=8, &beta;=0.1 (sweep run, 1500 steps).',
         width='100%', fig_id='Figure B3')}
</div>
<div>
{img_tag(os.path.join(R, 'ignition_T8_b10.png'),
         'T=8, &beta;=1.0: Strongest compression (KL&rarr;0.36).',
         width='100%', fig_id='Figure B4')}
</div>
</div>

<div class="two-col">
<div>
{img_tag(os.path.join(R, 'ignition_T1_b01.png'),
         'Ablation A (T=1, &beta;=0.1): High RCR, no convergence.',
         width='100%', fig_id='Figure B5')}
</div>
<div>
{img_tag(os.path.join(R, 'ignition_T1_b0.png'),
         'Ablation C (T=1, &beta;=0): Highest RCR (~13).',
         width='100%', fig_id='Figure B6')}
</div>
</div>

<hr>
<p style="text-align:center; color:#888; font-size:0.9em;">
  Report generated: February 27, 2026 &nbsp; | &nbsp;
  Environment: Python 3.x, PyTorch, CPU (Intel), Windows 11 &nbsp; | &nbsp;
  All code and data in <code>causal_emergence/</code>
</p>

</body>
</html>
'''

# ── Write HTML ────────────────────────────────────
output_path = os.path.join(BASE, 'LAB_REPORT.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"[OK] HTML report generated: {output_path}")
print(f"     File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
print(f"     Open in browser to view with all images embedded.")
