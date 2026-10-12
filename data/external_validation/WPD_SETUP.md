# WebPlotDigitizer Setup — Meropenem Time-Kill Digitization

## What's Ready

| File | Location |
|------|----------|
| WebPlotDigitizer (built) | `C:\Users\indan\webplotdigitizer\` |
| Figure 2 full page | `data\external_validation\figures\fig2_full.png` |
| Panel A (40% ATCC-27853) | `data\external_validation\figures\fig2A_40pct_ATCC.png` |
| Panel B (2x20% ATCC-27853) | `data\external_validation\figures\fig2B_2x20pct_ATCC.png` |
| Panel C (3x13.3% ATCC-27853) | `data\external_validation\figures\fig2C_3x13pct_ATCC.png` |
| CSV converter | `data\external_validation\wpd_to_csv.py` |

## Step-by-Step Workflow

### 1. Launch WebPlotDigitizer

```cmd
npx http-server C:\Users\indan\webplotdigitizer -o -c-1
```

This opens your browser at `http://localhost:8080`. Click **dev.html** or navigate to:
```
http://localhost:8080/dev.html
```

### 2. Load a Panel Image

For each panel (A, B, C):
1. Click **Load Image** (or drag-and-drop)
2. Select the panel PNG from:
   `data\external_validation\figures\fig2A_40pct_ATCC.png`
   (repeat for B and C)

### 3. Calibrate Axes

1. Click **Axes Calibration** → **2D (X-Y) Plot**
2. Click 4 known points on the axes:
   - **Point 1:** origin (time=0, lowest y value ≈ log10 CFU/mL = 1)
   - **Point 2:** right edge (time=24, lowest y value)
   - **Point 3:** origin again (time=0, same as point 1)
   - **Point 4:** top (time=0, highest y value ≈ log10 CFU/mL = 10)
3. Enter the known values when prompted:
   - X1=0, X2=24
   - Y1=1, Y2=10 (log10 CFU/mL scale)

### 4. Extract Data Points

For each concentration series (4xMIC and 16xMIC separately):

1. Click **Automatic Algorithms** → **X Step w/ Interpolation** or use **Manual Mode**
2. **Manual mode** (recommended for scatter plots):
   - Click **Manual Mode**
   - Click each data point on the curve
   - Each click adds (time, log10 CFU/mL) to the dataset
3. For 4xMIC: click all data symbols for the 4xMIC curve
4. Export: **Data Points** → **Download .txt** → save as e.g. `fig2A_4xMIC.txt`
5. Repeat for 16xMIC → save as `fig2A_16xMIC.txt`

### 5. Convert to CSV

After exporting WPD data files, run:

```cmd
cd "C:\Users\indan\OneDrive - aiimsbhubaneswar.edu.in\QSP_Antibiotics\data\external_validation"

:: Panel A — 40% regimen
python wpd_to_csv.py fig2A_4xMIC.txt "40%" "4xMIC"
python wpd_to_csv.py fig2A_16xMIC.txt "40%" "16xMIC"

:: Panel B — 2x20% regimen
python wpd_to_csv.py fig2B_4xMIC.txt "2x20%" "4xMIC"
python wpd_to_csv.py fig2B_16xMIC.txt "2x20%" "16xMIC"

:: Panel C — 3x13.3% regimen
python wpd_to_csv.py fig2C_4xMIC.txt "3x13.3%" "4xMIC"
python wpd_to_csv.py fig2C_16xMIC.txt "3x13.3%" "16xMIC"
```

This merges the digitized points into `meropenem_timekill_filled.csv`, replacing NA values.

### 6. Verify

Open `meropenem_timekill_filled.csv` and check that:
- No `NA` values remain
- t=0 values are ~6.176
- t=24 values match the paper's reported endpoints
- Intermediate values show expected kill/regrowth patterns

## Tips

- **Zoom in** (scroll wheel) before clicking data points for precision
- **Color filter** in WPD can help isolate specific symbol colors
- Save your WPD project (File → Save Project) between sessions
- The paper uses: ● = GC centrifuged, ○ = GC native, ★ = MIC
- For each panel, digitize 4xMIC and 16xMIC curves separately
