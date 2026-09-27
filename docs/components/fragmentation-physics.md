# Fragmentation Physics — First Principles for Tandem MS (ESI CID/HCD, timsTOF)

Scope: small-molecule (< ~1500 Da) tandem mass spectrometry as used in this competition —
electrospray ionisation (ESI), positive/negative mode, beam-type collision-induced
dissociation (CID) / higher-energy collisional dissociation (HCD), timsTOF-class
quadrupole–time-of-flight instruments. No code here; this is the physics the scoring
channels (library match, analog propagation, MetFrag-lite, formula/mass filters) are all
approximating.

---

## 1. What a tandem-MS spectrum actually is

A tandem-MS (MS/MS) experiment has three stages:

1. **Ionise** the neutral molecule M into a gas-phase ion (e.g. [M+H]+, [M+Na]+, [M−H]−).
2. **Isolate** one precursor m/z window (the intact ion, plus isotopes/adducts nearby).
3. **Activate** the isolated ion with collisions against inert gas (N2, Ar) and **record the
   masses of the charged pieces** that fall out.

The detector sees only ions. Neutral pieces are invisible — inferred by mass difference
(neutral loss = precursor mass − fragment mass). A fragment peak therefore encodes exactly
one number with high precision (m/z of a charged substructure, often < 5 ppm on TOF) plus
one rough number (its intensity). Everything else — which atoms the fragment contains, where
the charge sat, whether rearrangement occurred — must be inferred. That asymmetry (precise
mass, ambiguous structure) is the whole identification problem.

Energy scale matters: a covalent bond is ~3–5 eV (~300–500 kJ/mol). One collision with N2
at 10–40 eV lab energy deposits a fraction of an eV into internal modes; after tens to
hundreds of collisions the ion accumulates several eV of vibrational energy and then
unimolecularly dissociates. Fragmentation is thus **statistical**: energy randomises over
all vibrational degrees of freedom first, then the weakest accessible channel wins. This is
RRKM / quasi-equilibrium theory in one sentence: the ion "forgets" how it was heated and
breaks at its weakest link, modulated by entropy (how loose the transition state is).

## 2. Ionisation and adduct formation — where the charge lives

### 2.1 Electrospray in one paragraph

ESI sprays charged droplets; solvent evaporates, Coulomb repulsion blows the droplets
apart, and desolvated ions remain. For neutral natural products the dominant positive-mode
event is **protonation** ([M+H]+) or **cation adduction** ([M+Na]+, [M+K]+, [M+NH4]+);
in negative mode it is **deprotonation** ([M−H]−) or anion adduction ([M+Cl]−, [M+HCOO]−).
Which adduct forms depends on mobile-phase additives (formic acid favours H+; ubiquitous
sodium leaches from glassware), analyte functional groups, and concentration — not just the
molecule. The same neutral routinely appears as several adducts, each of which fragments
differently (Section 6).

### 2.2 Charge location: the single most predictive fact

Fragmentation chemistry is controlled by where the charge sits at the moment of cleavage:

- **Localised charge, charge-driven cleavage.** A proton on a carbonyl oxygen weakens the
  adjacent C–C/C–O/C–N bonds by withdrawing electron density; cleavage alpha to the
  charge site is accelerated by orders of magnitude. The heteroatom that "holds" the
  proton dictates which bonds are labilised. Basicity order in the gas phase (amines >
  most carbonyls > alcohols/ethers > esters) predicts protonation site, hence which end of
  the molecule fragments first.
- **Mobile-proton model.** With enough internal energy the proton migrates along
  hydrogen-bonding networks before fragmentation. A molecule with one strongly basic site
  and many weak sites therefore fragments at many positions as the proton samples them —
  each sampled position seeds a different cleavage series. Peptide chemists formalised this
  (Wysocki, Gaskell, Dongré): fragmentation diversity scales with proton mobility. Small
  molecules with several oxygens/nitrogens are mobile-proton systems par excellence.
- **Charge-remote fragmentation.** When the charge is fixed (quaternary ammonium, sodiated
  carboxylate, deprotonated acid that cannot mobilise) or the energy is very high, bonds
  break far from the charge through thermal/radical pathways — typically alkyl-chain
  cleavages with hydrogen transfers. These produce regular ladders (e.g. 14 Da CH2 series
  in lipids) that carry little substructure information per peak.

Rule of thumb: **mobile charge → few intense charge-driven cleavages near basic sites;
immobile charge → many weak charge-remote cleavages spread over the scaffold.** Both
regimes appear in one spectrum when adduct populations are mixed.

## 3. Why molecules break where they do

### 3.1 Bond dissociation energies (BDEs) — the first filter

Not all bonds are equal. Representative homolytic BDEs (gas phase, kJ/mol):

- C–I (~240) < C–Br (~285) < C–Cl (~330) < C–C (~350–370) ≈ C–N (~305–355) < C–O
  (~360–380) < C–H (~410–460) < C=C (~615, pi component ~270) ≈ C=O (~750).
- Allylic and benzylic C–C/C–H bonds are ~50–90 kJ/mol weaker than alkyl ones because the
  product radical/cation is resonance-stabilised. Alpha-heteroatom bonds (C–C next to
  O/N) are similarly weakened by lone-pair donation into the forming cation.
- Ring bonds cost extra: breaking one bond in a ring changes nothing observable (the ion
  stays intact); you must break **two** bonds to liberate a fragment, paying ~2× the
  energy for one mass signal. Polycycles are therefore intrinsically fragmentation-resistant.

CID deposits a few eV; only channels within roughly the bottom 1–2 eV of the accessible
barrier heights open. That is why spectra are sparse: out of dozens of bonds, only the
handful of weakest, charge-activated, entropically loose cleavages register.

### 3.2 The even-electron rule

ESI produces almost exclusively **even-electron ions** ([M+H]+ has a closed shell). The
rule: even-electron ions prefer to fragment into even-electron fragment + neutral molecule,
not radical + radical cation. Concretely, expect neutral losses of closed-shell molecules
(H2O, CO, NH3, HCOOH, CH3OH) and dislike radical losses (•CH3, •OH) — the latter appear
mainly at high energies or from odd-electron precursors (EI, not ESI). Violations flag
either high-energy channels or misassigned adducts. Practical consequence for scoring:
fragment masses should be explainable as subformula ± small H-shifts of the precursor
formula; radical-mass defects that break the nitrogen rule / electron parity deserve
suspicion.

### 3.3 The Stevenson rule (charge retention)

When a bond breaks heterolytically, A–B+ → A+ + B or A + B+, the charge stays with the
fragment of **lower ionisation energy** (better able to stabilise positive charge). For
positive mode: alkyl/aryl cations beat acylium beats oxonium beats ammonium, modulated by
resonance and substitution (tertiary > secondary > primary; conjugated > isolated). This
predicts which side of each cleavage you actually observe — the other side leaves as an
invisible neutral. Half of every fragmentation event is therefore systematically missing
from the spectrum, and which half survives flips with substitution patterns that barely
change the mass. Scoring implication: a candidate that explains the *neutral-loss ladder*
(precursor − peak) is as valuable as one that explains the peak as a charged substructure;
both are the same event viewed from opposite sides.

### 3.4 Privileged neutral losses — the usual suspects

Because even-electron, charge-driven chemistry funnels through a few low-barrier exits,
most MS/MS intensity sits in a short menu of neutral losses (positive mode):

- **H2O (18.0106):** alcohols, especially secondary/tertiary and alpha to unsaturation;
  cascades (18, 36, 54) count hydroxyls. Requires a proton near the OH and a neighbouring
  H — ubiquitous in polyols, sugars, polyketides.
- **CO (27.9949), CO2 (43.9898), HCOOH (46.0055):** carbonyls, acids, lactones, quinones;
  decarboxylation is nearly diagnostic of acids in negative mode ([M−H]− → −44).
- **NH3 (17.0265):** primary amines, amino acids; often competes with H2O loss.
- **CH3OH (32.0262), EtOH, larger alcohols:** methyl/ethyl esters and ethers.
- **162.0528 / 146.0579 / 132.0423 (hexose / deoxyhexose / pentose):** glycosidic
  cleavage — the single most intense event in most glycosides (Section 7).
- **Halogens:** HCl/HBr/HI losses (36/78–80/128) and halogen-isotope doublets (35/37Cl
  3:1, 79/81Br 1:1) are unmistakable flags.

A scoring function that recognises these losses by exact mass gets most of the explainable
intensity for free; a function that treats every dalton equally wastes discrimination on
noise.

## 4. Rearrangements — why prediction explodes combinatorially

Simple cleavage (cut one bond, keep one side) is the minority of pathways. The majority of
diagnostic peaks involve **rearrangement**: bonds break and form concertedly, atoms migrate.

- **McLafferty rearrangement:** carbonyl + gamma-hydrogen → enol ion + neutral alkene via
  a six-membered transition state. The fragment contains atoms that were never directly
  bonded in the precursor (the H moved). Mass-only models that cut bonds without moving Hs
  miss it unless they allow H-shifts — and allowing ±H on every cut multiplies candidates.
- **Retro-Diels–Alder (rDA):** cyclohexene → diene + dienophile. Two bonds break, one
  pi-bond reorganises; dominant in flavonoids, terpenes, alkaloids with fused rings. The
  fragment masses map to ring halves no single-bond cut can produce.
- **Hydride/alkyl shifts, ring openings, transesterifications:** under CID timescales
  (micro- to milliseconds) there is time for 1,2-shifts and ring–chain tautomerism before
  dissociation. Sugars ring-open and scramble; esters migrate; epoxides rearrange to
  carbonyls. Each reversible step branches the pathways.
- **Consecutive fragmentation:** primary fragments retain internal energy and fragment
  again (MS/MS is really MS^n collapsed into one spectrum at high CE). A peak may be two
  or three cleavages removed from the precursor, with each step's neutral loss ordering
  permutable (H2O-then-CO vs CO-then-H2O give the same mass but different intermediates).

Why this is combinatorially explosive: with N bonds, single cuts give O(N) fragments and
pairs O(N²). Add H-migration (±2 H per cut: ×5), rearrangement templates (×dozens), ring
opening before cleavage (each ring: open at any bond first), and sequential depth d
(branching factor b: b^d). A 40-bond natural product easily admits 10^4–10^6 mass-distinct
hypotheses within 10 ppm — most isobaric with each other. Mass accuracy prunes chemistry
only weakly: at 300 Da, ±5 ppm admits hundreds of elemental compositions, and each
composition admits many connectivity isomers. **Prediction is hard not because masses are
imprecise but because the forward map (structure → spectrum) is many-to-many**: many
pathways converge on the same mass, and one structure fans out to many masses depending on
energy, charge site, and history.

MetFrag-lite (single/double cuts, ±2 H, uniform bond cost) is therefore a deliberately
narrow slice: it captures direct glycosidic/ester/amide cleavages well and eats the full
loss on rearrangement-driven scaffolds (flavonoid rDA, terpene cascades). Its value is
orthogonality — it scores *this candidate's* bond graph with no library needed — not
completeness.

## 5. Charge retention vs migration, and adduct effects

The adduct determines the fragmentation regime more than the collision energy does:

- **[M+H]+:** mobile proton → rich charge-driven cleavage near basic sites, McLafferty and
  neighbouring-group losses, strong even-electron neutral-loss ladders. Most informative
  per peak; most library coverage.
- **[M+Na]+ / [M+K]+:** sodium binds hard to carbonyls/ethers (chelation) and does not
  migrate. Cleavage becomes charge-remote: higher barriers, fewer peaks, suppressed
  rearrangements, intact-cationised fragments ([fragment+Na]+). Spectra look "clean but
  empty" — high precursor survival, weak ladders. Cross-adduct matching ([M+H]+ library vs
  [M+Na]+ query) fails unless shifted by the adduct delta and rescored on neutral losses,
  because the *charged* fragments differ while the *neutral* losses partially transfer.
- **[M+NH4]+:** often dissociates first to [M+H]+ + NH3 (17 Da loss), then fragments as a
  protonated species — effectively a delayed proton source. Recognising the −17 precursor
  step unlocks the rest of the spectrum.
- **[M−H]−:** charge fixed on the most acidic site (carboxylate, phenolate); fragmentation
  is charge-driven around that site (decarboxylation −44, phenol-specific CO losses) plus
  charge-remote alkyl ladders. Acids/phenols fragment better in negative mode than
  positive; alkaloids/amines, the reverse. Polarity is a functional-group filter.
- **Dimers/multimers ([2M+H]+, [M+Cl]−):** fragment first to monomer at low energy; the
  monomer region then behaves normally. Unexplained high-mass peaks above M are usually
  dimers, not fragments.

Charge migration between fragments (proton-bound dimers that split either way, per the
Stevenson rule applied competitively) further means relative peak intensities encode
gas-phase basicity differences of the fragments — thermochemistry, not just structure.

## 6. Instrument dependence — timescale, energy regime, consecutive fragmentation

Same molecule, different instrument, different spectrum. Three axes:

1. **Energy deposition profile.** Beam-type CID/HCD (quadrupole collision cell, as in
   timsTOF and Orbitrap-HCD) accelerates ions through gas once: fast (microseconds),
   single-pass, deposits a broad internal-energy distribution; raising CE shifts the whole
   distribution up. Ion-trap CID instead resonantly excites the precursor slowly
   (milliseconds, many low-energy collisions): the precursor dissociates at threshold, and
   product ions fall out of resonance and go cold — **no consecutive fragmentation**
   (the "one-third rule" also cuts off low-m/z fragments). Consequence: beam-type spectra
   show deep ladders (fragments-of-fragments, rich low-mass region); trap spectra show
   first-generation fragments only. Library spectra from traps transfer poorly to
   beam-type queries at the low-mass end.
2. **Timescale vs rearrangement.** Traps give milliseconds for proton migration and ring
   opening before dissociation → rearrangement-rich. Beam-type gives microseconds →
   favours direct cleavage at high CE but compensates with consecutive fragmentation. The
   rearrangement/direct ratio is an instrument property, not a molecular one.
3. **Collision energy and ramp.** timsTOF-PASEF ramps CE with mobility/m/z; public libraries
   mix fixed-CE and ramped spectra. Intensity patterns shift systematically with CE
   (precursor survival down, low-mass ladder up), while peak *presence* is more stable.
   Scoring presence (explained-mass fraction) transfers across instruments better than
   scoring intensity (dot product); intensity-weighted scores need CE-matched references.

Bottom line for retrieval: match **same ionisation, same adduct, same polarity, same
instrument class, similar CE** first; back off to neutral-loss / shifted matching only when
those fail. Every relaxed dimension costs discrimination.

## 7. Why natural products are the worst case

Natural products combine every fragmentation-hostile property at once:

- **Polycyclic and fused rings.** Breaking a ring requires two concerted cleavages
  (Section 3.1); bridged/spiro systems need three. Barriers stack, so the molecular ion
  survives while the informative interior stays shut — spectra are sparse exactly where
  discrimination is needed. Ring-opening rearrangements then scramble the few fragments
  that do escape.
- **Glycosides.** The glycosidic bond is the weakest link by far, so ~all ion current
  funnels into sugar loss (162/146/132) plus aglycone — hundreds of distinct glycosides
  share the same aglycone peak and the same sugar loss. The discriminating detail (sugar
  identity, linkage position, anomerism) is carried by weak cross-ring cleavages that
  beam-type CID barely populates.
- **Oxygen tapestry.** Polyketides, tannins, and glycosides pack OH/ether/carbonyl groups
  that all lose H2O/CO with near-identical masses. Water cascades (−18−18−18) count
  oxygens but not positions: positional isomers (OH at C3 vs C7) give the same ladder.
- **Halogens and heteroatoms.** Marine/bacterial NPs carry Cl/Br whose isotope patterns
  dominate similarity scores without identifying the scaffold; S/P/B-containing metabolites
  fall outside standard fragment-element tables.
- **Stereochemical density.** NPs are chiral (many stereocentres, E/Z, atropisomerism) and
  CID is essentially blind to stereochemistry (gas-phase diastereomeric transition states
  differ by < 1 kJ/mol; intensities shift slightly, masses not at all). Ion mobility (TIMS)
  separates some isomers by shape — the one orthogonal axis CID lacks.

## 8. What a fragment peak fundamentally carries vs what is irrecoverable

**Recoverable from good MS/MS (in principle):**

- Elemental composition of precursor (exact mass + isotope pattern + adduct arithmetic).
- Presence of labile groups (OH count from water cascade, acid from −44 in negative mode,
  amine from −17, glycosylation from 162/146/132, halogen count from isotope ratios).
- A bag of subformulae (charged fragments + neutral losses) constraining connectivity —
  enough to rank same-mass candidates when scaffolds differ (this is the MetFrag signal).
- Scaffold class when a rearrangement is diagnostic (flavonoid rDA halves, prenyl 56/68
  losses, nucleoside base losses).

**Irrecoverable from CID MS/MS alone (no matter the resolution):**

- **Positional isomerism** on symmetric or repetitive scaffolds (which ring carbon bears
  the OH; which sugar hydroxyl links) — identical fragment masses, near-identical
  intensities.
- **Stereochemistry** — enantiomers are MS-identical; diastereomers differ only in weak
  intensity ratios, below library-transfer noise.
- **Charge-invisible halves** — per the Stevenson rule, every observed fragment implies an
  unobserved neutral; two candidates differing only on the neutral side are
  indistinguishable from that peak.
- **Rearrangement history** — McLafferty/rDA/shift products report the rearranged
  connectivity, not the precursor's; inverting the rearrangement from mass alone is
  underdetermined (many precursors, one product mass).
- **Intensities as thermometers** — intensities encode CE/instrument/pressure as much as
  structure; they transfer only between matched conditions.

Hence the competition's metric reality: exact-structure (InChIKey) retrieval from MS/MS
tops out where positional/stereo isomers begin, and the winning strategy is
*retrieval + orthogonal evidence* (formula, fingerprints, taxonomy, elution/mobility)
rather than fragmentation modelling alone.

## 9. Why fragmentation is hectic — the causal chain

> Ionisation parks one charge wherever the molecule is most basic (or chelates hardest) →
> collisional heating randomises that energy over every vibration before anything breaks →
> the mobile proton samples sites while barriers compete, so one precursor fans out over
> direct cleavages, H-shifts, ring openings, and rearrangements simultaneously → each
> primary fragment stays hot and fragments again, permuting loss order → the charge keeps
> only the lower-ionisation half of every split (Stevenson), hiding the other half as
> neutrals → the detector records precise masses of an unknown subset of an exponentially
> branching, history-dependent, adduct- and instrument-specific pathway tree, with
> positional and stereo isomers projecting onto nearly identical mass sets.

Hectic is not noise — it is **deterministic chemistry observed through a lossy projection**
(mass of charged survivors only). Scoring works to the extent it inverts the survivable
part of that projection: exact subformula masses, privileged neutral losses, adduct-aware
shifts, and instrument-matched intensity patterns — while refusing to hallucinate the
irrecoverable remainder.
