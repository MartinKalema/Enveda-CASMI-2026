# Fragmentation Sub-Problems — Shelf Coverage, Gaps, New Papers (no commit)

Scope: map every paper on `docs/papers/fragmentation/` to the 10 fragmentation
sub-problems, find the ZERO-coverage gaps, fill them with new papers, and name
the unanswered sub-problem that most limits exact-structure ranking.
Companion docs: `fragmentation-sota.md` (per-paper teardowns + scorer designs),
`frag-gaps.md` (kernel gaps G1–G10), `fragmentation-physics.md` (first principles).
Do NOT commit — analysis only.

Corpus after this task: 9 arXiv PDFs + 7 PMC JATS XMLs + `txt/` extracts, all
`file`-verified. (`fragmentation-sota.md` §1 lists 8 arXiv + 5 PMC = 13; the
shelf additionally holds `arxiv-2307.08240.pdf` = MIST-CF, mapped below but not
re-torn-down. Plus 2 new PMC papers added here: PMC6099382, PMC11492807.)

## 1. The 10 sub-problems

| # | Sub-problem | Question it answers |
|---|---|---|
| S1 | which-bonds-break / BDE | Which bonds cleave? Bond-strength ordering of the search |
| S2 | neutral losses | Which closed-shell NLs (H2O, NH3, CO, …) carry the intensity? |
| S3 | rearrangements | McLafferty / rDA / H-shifts beyond direct cuts |
| S4 | charge retention / Stevenson | Which side of each cleavage keeps the charge (and is observed)? |
| S5 | intensity prediction | Peak heights, not just peak presence |
| S6 | consecutive fragmentation | Fragments-of-fragments (depth > 1, MS^n collapsed in one spectrum) |
| S7 | adduct effects on fragments | Na+/K+/NH4+/formate behaviour at fragment (not precursor) level |
| S8 | CE / instrument dependence | How spectra shift with collision energy / instrument class |
| S9 | ring cleavage | Ordered 2-in-same-ring cuts (1 cut in a ring = intact mass) |
| S10 | H-rearrangement counts | Quantitative H-shift distributions per cleavage (±H ladder) |

Criterion: ● = paper models/measures/studies it; ◐ = touched (feature,
mention with numbers, conditioning variable); ○ = absent. Mere background
mention does not earn ◐.

## 2. Coverage matrix

| Paper | S1 BDE | S2 NL | S3 rearr | S4 charge | S5 inten | S6 consec | S7 adduct | S8 CE/inst | S9 ring | S10 H |
|---|---|---|---|---|---|---|---|---|---|---|
| CFM-ID 2014 (Allen, PMC4086103) | ● | ● | ○ | ○ | ● | ● | ○ | ◐ | ○ | ○ |
| CFM-ID 4.0 (Wang, PMC9064193) | ● | ● | ◐ | ◐ | ● | ● | ● | ● | ● | ◐ |
| MetFrag 2.2 (Ruttkies, PMC4732001) | ● | ● | ● | ○ | ◐ | ● | ● | ◐ | ◐ | ○ |
| Frag trees reloaded (Böcker, 1412.1929) | ○ | ● | ○ | ○ | ○ | ● | ○ | ○ | ○ | ○ |
| NEIMS (Wei, 1811.08545, EI) | ◐ | ● | ○ | ○ | ● | ○ | ○ | ○ | ○ | ○ |
| MassFormer (Young, 2111.04824) | ◐ | ◐ | ◐ | ○ | ● | ○ | ○ | ◐ | ○ | ○ |
| GrAFF-MS (Murphy, 2301.11419) | ○ | ● | ◐ | ○ | ● | ◐ | ● | ◐ | ○ | ○ |
| SCARF (Goldman, 2303.06470) | ◐ | ● | ● | ○ | ● | ○ | ◐ | ○ | ◐ | ◐ |
| ICEBERG (Goldman, 2304.13136) | ● | ● | ● | ○ | ● | ● | ◐ | ○ | ◐ | ● |
| MIST-CF (Goldman, 2307.08240) | ○ | ◐ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ○ |
| FraGNNet (Young, 2404.02360) | ● | ● | ● | ○ | ● | ● | ○ | ◐ | ◐ | ● |
| MassSpecGym (Bushuiev, 2410.23326, benchmark) | ○ | ◐ | ○ | ○ | ◐ | ○ | ○ | ◐ | ○ | ○ |
| FIORA (Nowatzky, PMC11889238) | ● | ● | ● | ○ | ● | ○¹ | ◐ | ● | ◐ | ● |
| SingleFrag (Pérez-Ribera, PMC12245663) | ○ | ○ | ○ | ○ | ◐² | ○ | ○ | ◐ | ○ | ○ |
| **NEW De Vijlder tutorial (PMC6099382)** | ◐ | ● | ● | ● | ◐ | ● | ◐ | ◐ | ○ | ● |
| **NEW Lee CIDMD protomers (PMC11492807)** | ◐ | ○ | ◐ | ● | ○ | ○ | ○ | ◐ | ○ | ○ |

¹ FIORA is explicitly single-step only (no cascades) — S6 ○ by design.
² SingleFrag argues presence > intensity — S5 ◐ as deliberate counter-evidence.

Column tallies (● count, before → after): S1 5→5, S2 10→11, S3 5→6, **S4
0→2**, S5 9→9, S6 6→7, S7 3→3, S8 2→2, S9 1→1, S10 3→4.
(◐-only papers excluded from ● tallies: CFM-4.0's S4/S10 stay ◐.)

## 3. Gap list

- **S4 charge retention / Stevenson — ZERO ● on the shelf (the gap).** Best
  prior touch: CFM-ID 4.0 asks "which fragment will have a charge" with
  Gasteiger-charge + atom-type features (◐). Everything else either assumes
  the kept fragment is charged (ICEBERG Generate "needs not predict or track
  whether structures are charged"; SCARF/FraGNNet/FIORA reduce to neutral
  formulae + adduct shift) or never asks the question. Zero mentions of
  Stevenson, Field's rule, mobile proton, even-electron rule, CRF/CMF in any
  of the 14 shelf texts (verified by grep).
- Near-gaps (thin, NOT zero — no new papers needed): S9 ring cleavage (1 ●:
  CFM-4.0 sequential cleavage), S8 CE/instrument (2 ●: FIORA continuous-CE
  covariate, CFM-4.0 per-energy models), S7 adduct-on-fragments (3 ●:
  MetFrag-2.2 product ions, CFM-4.0 adduct rules, GrAFF-MS adduct states).
- Everything else (S1/S2/S3/S5/S6/S10) has ≥3 ● — no gap.

## 4. New papers (both `file`-verified JATS XML + `txt/` extracts)

**N1 — De Vijlder et al., "A tutorial in small molecule identification via
ESI-MS: the practical art of structural elucidation", Mass Spectrom Rev
2017 (review), `PMC6099382.xml` (266 KB).** What it adds (all previously
absent from the shelf): charge-directed vs charge-remote fragmentation
(Cheng & Gross pointer); inductive cleavage = C–heteroatom cleavage with
charge migration to α-carbon (CMF) vs same-bond cleavage with charge
retention on the heteroatom by proton rearrangement (CRF) — with the
operational **[M+H]⁺+1 complement rule** (both fragments sometimes observed,
nominal m/z sum = precursor+1); C–C α-to-heteroatom cleavage retaining
charge as iminium/oxonium; even-electron/parity rule + N-rule extension +
radical-loss exceptions (halogen/nitro from polyaromatics); Tsugawa's nine
H-rearrangement rules (C/N/O/P/S, both polarities) as the compact encoding
of most other rules; beam-type second-generation products (haloperidol
m/z 165 → 194 via −H2O: consecutive fragmentation made concrete);
Weissberg & Dagan rule list + Niessen class-specific reviews as the
rule-mining backlog; solution- vs gas-phase basicity for the ionisation
site stated as *unresolved*. Kernel reading: the complement rule + CRF/CMF
split is the rule-level spec for frag-gaps G7 (retention prior) — and the
basicity-unresolved verdict says a learned protonation-site term will not
transfer without calibration data.

**N2 — Lee et al., "Impact of protonation sites on CID-MS/MS using CIDMD
quantum chemistry modeling", J Chem Inf Model 2024, doi
10.1021/acs.jcim.4c00761 (companion to 4c00760 CIDMD framework),
`PMC11492807.xml` (126 KB via eutils efetch; EuropePMC fullTextXML 500'd,
BioC endpoint 429/no-result).** 10 small molecules × 43 protomers, CIDMD
(1 ps trajectories) vs all NIST20 experimental spectra. What it adds:
(1) **proton affinity does NOT predict which protomer fragments** — 4/10
compounds show strong *negative* PA↔similarity correlation; the highest-PA
protomer (e.g. taurine N-protonated) scores worst; (2) different protomers
give widely different spectra yet converge on the same product ions via
different pathways — single-protomer simulation is structurally
insufficient (uracil: only one protomer explains all ions, but with
false positives); (3) proton-transfer barriers are low-eV, accessible
during ESI (250–350 °C, >3 kV) — direct computational support for the
mobile-proton model in *small* molecules, possibly water-assisted;
(4) prescriptive conclusion: simulate a protomer *mixture* weighted by
pathway bottleneck free energy, not Boltzmann populations of the intact
ion. Kernel reading: kills the "protonate the most basic site and fragment
from there" shortcut (frag-gaps G7's N>O>S proxy is exactly the flawed
prior); any retention model must be a *mixture over protomers*, and
intensities encode fragment gas-phase basicity differences (Field's rule),
not just structure.

## 5. Which unanswered sub-problem most limits exact-structure ranking

**S4 charge retention — by a distance.** Three reasons:

1. **It corrupts 100% of scoring events, not a tail.** Every cleavage
   produces one observed ion + one invisible neutral (fragmentation-physics
   §3.3, §8). MetFrag-lite scores *both* sides as explainable (frag-gaps
   G7), so decoys collect free hits on neutral-side masses — compressing
   the truth/decoy gap exactly where ranking lives (isomer MRR frag 0.28
   vs analog 0.54). BDE ordering (S1) or NL ladders (S2) add *more* masses;
   only S4 removes *wrong* ones.
2. **It is the isomer discriminator of last resort.** Positional isomers
   (OH at C3 vs C7) give identical fragment-mass sets; what differs is
   *which side keeps the charge and with what intensity* (Stevenson/Field's
   rule = fragment basicity competition, N2 §4). No shelf model predicts
   this — all learned scorers inherit charge assignment from training-data
   correlation, which is precisely what fails on MCES-novel scaffolds.
3. **The fix is kernel-cheap, the data is not.** A CRF/CMF split with the
   [M+H]⁺+1 complement rule (N1) + protomer-mixture retention prior (N2) is
   ~0 CPU (one basic-site tag per fragment, cf. frag-gaps G7). What is
   missing is calibration: measured which-side-keeps-charge labels per
   cleavage class, i.e. annotated spectra with complementary-ion pairs —
   no shelf dataset provides them (MassSpecGym annotates formulae, not
   charge sides; FraGNNet's P(node|formula) is the closest and still
   charge-blind).

Runner-up: S9 ring cleavage (96% of NP candidates carry rings; single ●).
But S9 is at least *formulated* (ordered 2-cuts) in CFM-4.0 and ported in
Design A; S4 has no learned treatment anywhere and biases every candidate
score. Fix priority for ranking: S4 retention prior first (kills decoy
inflation), S9 enumeration second (recovers missing masses).

*Do NOT commit — analysis only.*
