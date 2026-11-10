# Mesh design: NACA 0012 wingtip vortex, SST vs SST-CC

The mesh is built to compare k-ω SST and SST with curvature correction (SST-CC) against Chow, Zilliac & Bradshaw (1997). That comparison only means something if the numerical diffusion in the vortex core is small next to the modelled (eddy-viscosity) diffusion, and if the boundary layer that feeds the vortex, over the rounded tip, is resolved to the wall. Everything below follows from those two requirements.

All numbers in §6 and §7 come from meshes actually generated with these files (snappyHexMesh v2312, 8 MPI ranks). The dictionaries use only v2306-compatible syntax.

Files changed or added:

| File | Change |
|---|---|
| `OpenFOAM/system/blockMeshDict` | Same domain and patches. Background cells stretched 2:1 in x; cell counts are the grid-study parameter (§4, §7) |
| `OpenFOAM/system/snappyHexMeshDict` | Rewritten (§5) |
| `OpenFOAM/system/surfaceFeatureExtractDict` | Keeps only the trailing-edge corner edges |
| `OpenFOAM/system/controlDict` | Steady `simpleFoam`; adds `yPlus`, `wallShearStress`, `residuals`, `forceCoeffs` function objects |
| `geometry/build_wing_stl.py` | New. Builds the wing STL analytically (§5.4) |
| `OpenFOAM/constant/triSurface/wing.stl` | Output of the script above (binary STL) |

Also changed for steady RANS: `constant/transportProperties` (ν matched to Re_c), `system/fvSchemes`, `system/fvSolution`, `system/controlDict` (`simpleFoam`, iteration-based, residual and force-coefficient monitors). `0/nut` uses `nutUSpaldingWallFunction` on the wing (§2). Other boundary conditions and `meshQualityDict` are unchanged.

---

## 1. Assessment of the existing mesh

Cell size at level *n* is Δ₀/2ⁿ with Δ₀ = 203.2 mm: L3 25.4, L4 12.7, L5 6.35, L6 3.18, L7 1.59, L8 0.79 mm.

**Vortex core: the main limitation.** The measured core radius r_c (radius of peak crossflow velocity, `TAKALL.DAT`) is about 14 mm at x/c = 0.44 over the wing, 30 mm at the trailing edge and 35–40 mm downstream. The old mesh puts 12.7 mm cells (L4) on the whole vortex path, which gives **2–6 cells across the core diameter**. A RANS vortex needs roughly 15–20 or more. At this resolution the numerical diffusion is as large as, or larger than, the model diffusion. That is consistent with the existing result (core Cp ≈ −1.8 against −3.6 measured), and it would hide most of the SST/SST-CC difference.

**Effort spent in the wrong place.** The L4 box (2.03 × 0.61 × 1.02 m) holds about 614 k of the ~700 k cells, spread uniformly, mostly in freestream. It also ends at x = 1.626 m, while the last measured plane is at x = 1.727 m. So the last station sits in 25–50 mm cells with a level change between stations 9 and 10.

**Wing surface.**
- Surface L4–5 gives 6–9 cells around the leading-edge semicircle (r_LE = 19 mm).
- The blunt trailing edge is 2.54 mm thick against 6.35 mm cells. It cannot be snapped, so the Kutta condition, and with it circulation and vortex strength, depends on how the mesh breaks down at the TE.
- The `features` entry asks for level 4, below the surface level, so it has no effect.

**Near wall.** First layer 0.38 mm (y⁺ ≈ 27–55) with high-y⁺ wall functions. The vortex starts as a crossflow separation off the rounded tip (radius of curvature 20–73 mm). That is a non-equilibrium, strongly curved boundary layer where the log law does not hold, and where curvature correction itself acts.

**Geometry (STL).** The gmsh export is too coarse for a wall-resolved mesh:
- On the rounded tip, neighbouring facets meet at up to 38° (median 6°, p90 14°).
- The facets miss the true surface by up to 0.7 mm on the wing and 1.4 mm on the tip.
- There is an open 2.5 mm half-disc at the tip trailing edge, and the tip/wing seam is non-conforming.

**Smaller issues.**
- `maxGlobalCells 2e6` would truncate any larger mesh part-way through a level.
- `locationInMesh (-1.2192 0 0.6096)` lies exactly on a background-mesh vertex.

**Viscosity (fixed).** `constant/transportProperties` had ν = 1.0e-5 m²/s, which gives Re_c = 6.3×10⁶. It is now ν = U∞·c/Re_c = 51.816 × 1.2192 / 4.6×10⁶ = **1.37335e-5 m²/s**, matching the experiment.

---

## 2. Recommended near-wall treatment

**Wall-resolved (low-Re) SST, first-cell y⁺ ≈ 1.**

- Curvature correction changes production through the rotation and strain-rotation terms. These are largest in the vortex core and in the strongly curved boundary layer around the tip, where the vortex is fed. Resolving that boundary layer down to the wall is what lets the two models differ there.
- The tip-vortex origin is a crossflow separation. Its location depends on wall shear and adverse pressure gradient, which wall functions model through an equilibrium log law that does not hold there.
- The published SST-CC/SARC studies of this case (e.g. Dacles-Mariani et al. 1995; Churchfield & Blaisdell 2009) use y⁺ ≤ 1 grids.

**ν_t wall condition: `nutUSpaldingWallFunction`** (changed from `nutkWallFunction`). `k` (`kqRWallFunction`) and `ω` (`omegaWallFunction`, which blends viscous and log-layer values) are unchanged.

The original `nutkWallFunction` estimates y⁺ from k in the first cell (C_μ^¼·y·√k/ν). Where that estimate is above y⁺_lam = 11.5, it applies a log-law ν_t at the wall and reports that k-based y⁺, even with a 10 µm first cell. Because `kqRWallFunction` is zero-gradient, near-wall k is not forced to zero, so this can happen on fine faces (e.g. near the LE stagnation line). The first server run (nutk, iteration 500) gave area-averaged y⁺ = 1.68 but a maximum of 251, which is impossible on faces with the full layer stack and points to this switch.

Spalding's law is continuous over the whole y⁺ range and is driven by the wall velocity gradient, not by k. It behaves as a low-Re condition at y⁺ ≈ 1, handles the ~4 % of faces with fewer layers gracefully, and makes the `yPlus` output the true wall-shear y⁺. It applies identically to SST and SST-CC.

---

## 3. y⁺ and first-layer thickness

Schlichting turbulent skin friction, C_f = (2 log₁₀ Re_x − 0.65)^−2.3, with local edge velocity U_e. ν = 1.37335e-5 m²/s, U∞ = 51.82 m/s.

| Location | U_e/U∞ | C_f | u_τ (m/s) | Δy for y⁺_centre = 1 | y⁺_centre with Δy₁ = 10 µm |
|---|---|---|---|---|---|
| Flat-plate mean (x = c) | 1.0 | 0.0029 | 1.97 | 13.9 µm | 0.72 |
| Mid-chord, suction side | 1.25 | 0.0031 | 2.56 | 10.7 µm | 0.93 |
| Tip, under the vortex | 1.8 | 0.0028 | 3.49 | 7.9 µm | 1.27 |
| LE suction peak (x/c ≈ 0.03) | 2.4 | 0.0046 | 5.96 | 4.6 µm | 2.2 |

**First layer: Δy₁ = 1.0×10⁻⁵ m.** This gives first-cell y⁺ ≈ 0.7–1 over most of the wing, ≈ 1.3 under the vortex and ≤ ≈ 2 in the narrow leading-edge suction peak. All of these are inside the viscous sublayer. Boundary layer: δ₉₉ ≈ 21 mm at the TE (flat plate; thicker on the suction side), δ⁺ ≈ 3000.

**Layer stack.** 25 layers. First layer absolute (10 µm), final layer = 0.3 × local cell size (OpenFOAM's `firstAndRelativeFinal` model, selected by giving exactly `firstLayerThickness` and `finalLayerThickness`). snappy picks the expansion ratio locally, so the prism-to-hex size jump is the same everywhere:

| Surface level | Where | Cell Δy | Final layer | Expansion | Stack height | y⁺ at top |
|---|---|---|---|---|---|---|
| 5 | inboard mid-chord | 6.35 mm | 1.90 mm | 1.245 | 9.7 mm | ~1400 |
| 6 | most of the wing | 3.18 mm | 0.95 mm | 1.21 | 5.5 mm | ~800 |
| 7 | LE, tip under vortex | 1.59 mm | 0.48 mm | 1.175 | 3.2 mm | ~450 |
| 8 | trailing edge | 0.79 mm | 0.24 mm | 1.14 | 1.9 mm | ~270 |

Measured on the medium mesh: stack height p10/p50/p90 = 3.1 / 5.5 / 9.7 mm, matching the table. The outer boundary layer is carried by L6/L5 hexes: the wing-distance refinement keeps L5 or finer up to 40 mm from the wall.

**Check after the first converged run:** the `yPlus` function object (now in `controlDict`) logs y⁺ min/avg/max on `wing`. Expect avg ≈ 0.5–1 and max ≈ 2–3. If the average is above 1.5, reduce `firstLayerThickness` to 7e-6.

---

## 4. Background mesh (`blockMeshDict`)

Domain (−96…144 × −16…16 × 0…48 in) and all six patches are unchanged. **Only the cell shape changes**: 16 × 8 × 8 in (2:1 in x) instead of 8 in cubes, i.e. 15 × 4 × 6 cells.

snappy splits cells isotropically, so every level keeps Δx = 2Δy = 2Δz:

- **Vortex:** its axis is within 1–6° of x. Radial gradients act over r_c = 15–40 mm, while streamwise changes act over about a chord. Core diffusion is controlled by the cross-plane (y,z) spacing, which is unchanged. Halving the streamwise count halves the cost of the vortex tubes.
- **Wing surface:** upper and lower surfaces have normals close to y, so their faces are Δx·Δz. Stretching in x halves the surface faces, and with them the prism cells, which are more than half of a y⁺ ≈ 1 mesh. Chordwise spacing at L6 is 6.35 mm (0.52 %c, ~190 cells per side), refined to L7/L8 at the LE and TE.
- Without this change the same resolution needs about 9–10 M cells.

---

## 5. Revised `snappyHexMeshDict`

### 5.1 Vortex path

The core was located on every measured crossflow plane (minimum Cp_static in `RETAKALL.DAT`, transformed with the repo's `transform_coords.py`). It was then compared with the core path of the previous SST solution (`scripts/results/plots/core_*_location.png`):

| x (m) | x/c from root QC | Measured y, z (m) | Previous SST y, z (m) | Measured r_c |
|---|---|---|---|---|
| 0.18 | 0.15 | −0.025, 0.916 | ≈ same | forming at tip edge |
| 0.54 | 0.44 | −0.035, 0.835 | −0.095, 0.89 | 14 mm |
| 0.66 | 0.54 | −0.057, 0.820 | −0.11, 0.875 | 20 mm |
| 0.89 | 0.73 | −0.096, 0.818 | −0.11, 0.845 | 30 mm (TE) |
| 1.20 | 0.98 | −0.081, 0.811 | −0.085, 0.793 | 38 mm |
| 1.73 | 1.42 | −0.068, 0.805 | −0.085, 0.808 | 38 mm (last plane) |

The vortex leaves the tip edge (z ≈ 0.915), moves inboard and down over the aft upper surface to z ≈ 0.82, then stays almost parallel to x. Experiment and old CFD differ by up to 55 mm over the wing. **The tube centrelines follow the midpoint of the two paths, and the radii cover the spread plus about 2 r_c.** Tapered `cone` searchable surfaces follow the growth of r_c.

### 5.2 Refinement regions

| Region | Level (Δy) | Geometry | Purpose |
|---|---|---|---|
| `vortexCoreFormation` | 7 (1.59 mm) | cone, x 0.10→0.54, r 30→55 mm | Core formation over the tip: r_c 10–20 mm → 18–25 cells across the core |
| `vortexCoreAft` | 7 | cone, x 0.53→0.92, r 56→50 mm | Aft upper surface to TE: r_c 20–30 mm → 25–38 cells |
| `vortexCoreNearWake` | 7 | cone, x 0.91→1.15, r 51→40 mm | Near wake, where the TE wake sheet rolls into the core |
| `vortexInnerWing` / `vortexInnerWake` | 6 (3.18 mm) | cones to x = 2.00, r 60–91 mm | ~2.5 r_c plus path spread. Downstream r_c ≈ 38 mm → 24 cells across the core. Ends 0.27 m (7 r_c) past the last plane |
| `vortexOuter` | 5 (6.35 mm) | cone to x = 2.25, r 120→150 mm | Outer swirl (v_θ ~ 1/r) and roll-up of the outboard wake sheet |
| `wakeSheet` | 4 (12.7 mm) | box x 0.80–2.50, z 0.30–0.99 | Wake sheet shed by the outboard wing that feeds the vortex |
| `nearField` | 3 (25.4 mm) | box around wing and wake | Potential flow and tunnel-wall interaction, gradual transition to the background |
| `wing` (distance) | 5 to 40 mm, 4 to 120 mm | — | Covers the whole boundary layer (δ ≈ 20–30 mm) outside the prisms |
| `wing` surface | 6, curvature → 7 | `curvatureLevel (10 0 7 -1)` | 10 cells per radius of curvature: L7 at the LE (r = 19 mm) and the tip/TE region |
| TE feature edges | 8 within 2 mm, 7 within 6 mm | `wing.eMesh` | ~3 cells across the 2.54 mm blunt TE: both corners snapped, layers on the TE face |
| `inboardMidChord` (limitRegions) | cap at 5 | box z < 0.50, −0.255 < x < 0.86 | Inboard half-span is nearly 2-D and far from the vortex. 12.7 mm chordwise (1 %c) is enough there. Removes ~1.5 M prism cells. LE and TE strips keep L7/L8 |

`nCellsBetweenLevels 3`: every 2:1 jump is followed by at least three cells of the same size, including across the vortex tubes.

**Core resolution, old vs new (cells across the core diameter 2r_c):**

| x (m) | r_c | Old | New |
|---|---|---|---|
| 0.54 | 14 mm | 2–4 | 18 |
| 0.66 | 20 mm | 3–6 | 25 |
| 0.89 (TE) | 30 mm | 5 | 38 |
| 1.20–1.73 | 38 mm | 6 (≤ 3 at x = 1.73) | 24 |

### 5.3 Layer addition: what was needed for 96 % coverage

Getting 25 layers with a 10 µm first cell onto a curved wing with snappy took three changes. Each one was found by generating the medium mesh and measuring per-face layer count and first-cell height:

| Run | Change | Faces with no layers | Faces with all 25 | Tip: mean layers | First cell p90 |
|---|---|---|---|---|---|
| 1 | defaults from `meshQualityDict` | 72 % | — | — | — |
| 2 | `minDeterminant 1e-6` | 17 % | 18 % | 1.0 | 1.4 mm |
| 3 | + analytic STL | 13 % | 24 % | 1.0 | 1.4 mm |
| 4 | + relaxed `minTetQuality` | 0.5 % | 42 % | 17 | 0.48 mm |
| **5 (final)** | + relaxed `minTwist` | **0.0 %** | **96 %** | **25** | **10.3 µm** |

1. **`minDeterminant 1e-6`** (override in `meshQualityControls`). snappy's scaled determinant is about 27·h²/(Δx·Δz) for a cell of height h. For h = 10 µm on 3–13 mm faces that is 3e-5 to 3e-4, so the default 10⁻³ rejects every near-wall prism. A low determinant here reflects the intended aspect ratio, not a defect.
2. **`relaxed { minTetQuality -1e30; minTwist -1; }` with `nRelaxedIter 0`** (layer phase only; snapping uses the full checks). On curved walls the snapped faces are slightly non-planar: a 3 mm face on a 20–70 mm radius is warped by about 0.1 mm, ten times the first layer. The face-centre tet decomposition and twist measure of those thin prisms then fail even though cell volumes and face pyramids are positive. With the full checks the tip kept layers on only ~1/3 of its faces.
3. **Why partial removal is so damaging.** With `nBufferCellsNoExtrude 0`, snappy tapers the layer count by one per ring of faces next to every un-extruded face and keeps the stack height, so tapered faces get a much thicker first cell. In run 4, 0.5 % un-extruded faces were enough to taper 58 % of the wing. Setting `-1` (no taper) gives abrupt layer ends, failing twist checks, and a segfault in v2312, so the fix has to be avoiding un-extruded faces in the first place.

Final coverage (medium mesh):

| Region | Faces | All 25 layers | Mean layers | First cell p50 / p90 |
|---|---|---|---|---|
| Whole wing | 102 k | 96.4 % | 24.7 | 10.0 / 10.3 µm |
| Leading edge (x/c < 0.05) | 16 k | 86 % | 23.9 | 10.0 / 17.5 µm |
| Trailing edge (x/c > 0.95) | 15 k | 100 % | 25.0 | 9.3 / 10.4 µm |
| Inboard mid-chord (z < 0.5) | 14 k | 95 % | 24.5 | 10.0 / 10.1 µm |
| Outboard mid-chord | 40 k | 98 % | 24.9 | 10.0 / 10.1 µm |
| Rounded tip | 18 k | 100 % | 25.0 | 10.0 / 10.2 µm |

### 5.4 Wing STL (`geometry/build_wing_stl.py`)

The script reproduces the geometry definition of `generate_wing.py`:
- same NACA 0012 equation;
- 0.100 in blunt TE: the profile is truncated where the half thickness reaches 0.05 in, then flat to x/c = 1;
- 33.12 in straight section;
- upper profile revolved 180° about the chord line;
- 10° rotation about the quarter chord, quarter chord at the origin.

It tessellates this finely:
- cosine chordwise spacing;
- 1.5° tip steps;
- conforming seam and closed TE half-disc.

Results:
- deviation from the analytic surface < 0.02 mm (gmsh export: up to 1.4 mm);
- facet kinks ≤ 1.5° (was up to 38°);
- watertight except the root, which lies on the symmetry plane;
- bounding box identical to the gmsh STL within 0.01 mm.

### 5.5 Other settings

- `maxGlobalCells 2e7`, `maxLocalCells 2e6`.
- `locationInMesh (-1.1937 0.01733 0.6371)`: freestream, not on any face at any level.
- `writeFlags (scalarLevels layerSets layerFields)`: writes `cellLevel`, `nSurfaceLayers`, `thickness` and `thicknessFraction`, so refinement and layer coverage can be checked in ParaView.

---

## 6. Cell count and mesh quality (medium mesh, measured)

| Part | Cells |
|---|---|
| Levels 0–4: freestream, near field, wake sheet | 175 k |
| Level 5: wing boundary-layer band, outer vortex tube, inboard surface | 337 k |
| Level 6: wing surface, inner vortex tube | 685 k |
| Level 7: vortex core cones, LE, tip | 747 k |
| Level 8: trailing edge | 14 k |
| Prism layers (102 k wing faces × ~24.7) | 2 521 k |
| **Total** | **4.48 M** |

So 2.0 M cells hold the volume resolution (of which ~1.4 M are on the vortex) and 2.5 M are wall-resolution layers. The 2 M budget cannot hold a y⁺ ≈ 1 mesh: the layers alone exceed it. The medium mesh is the smallest that resolves both the core (≥ 18 cells across) and the wall (y⁺ ≈ 1 with 25 layers).

`checkMesh` (default checks): **Mesh OK.**

| Metric | Value |
|---|---|
| Max skewness | 2.50 |
| Max aspect ratio | 808 (prism layers) |
| Min cell volume | 2.2e-12 m³ (positive) |
| Face pyramids | OK |
| Max non-orthogonality | 87.2°, average 9.6° |
| Faces > 70° | 16 795, **all inside the prism layers** (owner cell < 0.44 mm from the wall, median 0.04 mm). Outside the layers the max is ~68° |

`checkMesh -allGeometry` additionally flags face-tet quality (51 k faces), small determinant (< 10⁻³ in 1.47 M cells) and concave cells (71 k). All of these come from the thin layer cells on curved walls (§5.3). They affect particle tracking and `cellPoint` interpolation, not the finite-volume solution. The probes use the default `cell` interpolation.

**Solver settings for this mesh.** Because of the non-orthogonal layer faces:
- `snGradSchemes { default limited 0.5; }` and `laplacianSchemes { default Gauss linear limited 0.5; }` (now set);
- 1 non-orthogonal corrector (now set; raise to 2 if p stalls).

---

## 7. Grid-independence strategy

All snappy levels are relative to the background, and the layer stack is defined by an absolute first layer and a relative final layer. So **the only thing that changes between meshes is the three cell counts in `blockMeshDict`**:

| Mesh | nx, ny, nz | Background (in) | Wing faces | Cells before layers | Total cells | Status |
|---|---|---|---|---|---|---|
| Coarse | 12, 3, 5 | 20.0 × 10.7 × 9.6 | 69 k | 0.9 M | **2.34 M** (measured) | Generated; mean 19.4 layers per face, and `checkMesh` reports 1 failed check, **not yet investigated** |
| Medium | 15, 4, 6 | 16.0 × 8.0 × 8.0 | 102 k | 1.96 M | **4.48 M** (measured) | Generated and verified (§5.3, §6) |
| Fine | 19, 5, 8 | 12.6 × 6.4 × 6.0 | 225 k | 4.0 M (measured) | **~9.5 M** (estimate: + 225 k × ~24.5 layers) | Castellated stage only; layers not yet run |

Effective refinement ratios from the measured counts: coarse→medium (4.48/2.34)^(1/3) = 1.24, medium→fine ≈ (9.5/4.48)^(1/3) = 1.28.

The fine mesh has 2.2× the wing faces of the medium mesh, more than the r² ≈ 1.6 expected. Its layer count, and so its total, must be confirmed by a full run.

- First-layer thickness stays at 10 µm on all meshes (y⁺ < 1 already). Layer count stays at 25, and the expansion ratio drops automatically on finer meshes because the final layer follows the local cell.
- Effective refinement ratio r = (N_fine/N_medium)^(1/3). The Celik et al. (2008) GCI procedure prefers r > 1.3; 1.26 is acceptable and is the practical limit given the 3–5 cell background height.
- Monitored quantities, per measured plane: core Cp_static, peak crossflow velocity, r_c, axial velocity at the core, and core y, z (from `core_tracking.csv`). Also C_L, C_D and surface Cp at the `CHORDQPL.DAT` stations.
- Run **both** SST and SST-CC on all three meshes. The model difference in core Cp should be larger than the medium-mesh GCI; otherwise the comparison is mesh-limited.
- Cheaper core-only check: keep the medium mesh and raise the three `vortexCore*` cones from level 7 to 8 (and `vortexInner*` from 6 to 7). This isolates numerical core diffusion at about +1.5–2 M cells.

Hardware: on the 62 GB / 12-core server all three meshes fit for meshing and for `simpleFoam` (fine mesh ~15–20 GB). Run 12 MPI ranks (physical cores) with `mpirun -np 12 --map-by socket --bind-to core`, and keep the case off the full root partition.

---

## 8. Running and checking

```bash
python geometry/build_wing_stl.py OpenFOAM/constant/triSurface/wing.stl   # once (already done)
cd OpenFOAM
blockMesh
surfaceFeatureExtract
decomposePar -force
mpirun -np 12 --map-by socket --bind-to core snappyHexMesh -parallel -overwrite
reconstructParMesh -constant
rm -rf processor*
checkMesh
```

Medium mesh: about 26 min on 8 hardware threads; peak memory is modest.

In ParaView, check `cellLevel` (vortex tubes visible as level 6/7 bands) and `nSurfaceLayers` / `thicknessFraction` on `wing` (expect 25 and ≈ 1 almost everywhere, with a few LE faces lower).

## 9. Notes beyond the mesh (affect core diffusion just as much)

- Solver set-up is now steady `simpleFoam` (SIMPLEC) with the FSAE team `fvSchemes`/`fvSolution`, including `limited 0.5` non-orthogonal correction and 1 non-orthogonal corrector (§6).
- `div(phi,k)` and `div(phi,omega)` are first-order upwind (team standard, robust). Upwinding k and ω adds numerical diffusion to the turbulence quantities, including in the vortex core where SST and SST-CC differ. Once converged, consider re-running both models with `bounded Gauss linearUpwind grad(k)` / `grad(omega)` to confirm the model difference is not scheme-dependent.
