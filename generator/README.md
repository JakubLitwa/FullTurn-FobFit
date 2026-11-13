# FobFit generator

FobFit is a one-piece PETG jacket that snaps onto the plastic head of a car key. It adds a wing, a paddle or a finger ring, so a person with a weak pinch can turn the key with the side of the hand or a hooked finger.

This tool makes the jacket from a 3D scan of the key. The cavity follows the scanned shape, so the jacket holds an odd-shaped key by its form alone. It needs no screws, straps or liner. You need a scan only once per key model. The person who uses the key never needs a scanner.

## Install

1. Install Python 3.10 or newer from python.org. Tick "Add Python to PATH". The tool was tested with Python 3.13.
2. Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

3. Optional: install Bambu Studio. If the tool finds it, it slices the jacket in PETG for a Bambu Lab printer and reports the real mass and print time.

All the packages are free and have permissive licences (BSD, MIT or Apache). The window uses tkinter, which comes with Python.

**Without Python (Windows 10 or 11, 64-bit).** The Windows build is a `FobFit` folder with `FobFit.exe` and an `_internal` folder. It needs no Python. Copy the whole folder to Documents or the Desktop (not Program Files) and double-click `FobFit.exe`. Slicing still needs Bambu Studio. The exe is not signed, so the first start of a downloaded copy shows "Windows protected your PC": click More info, then Run anyway. To build it yourself, see `FobFit.spec`.

## Start

Double-click `FobFit.bat`. Or run:

```bash
python fobfit_gui.py
```

For a shortcut with the FobFit icon, double-click `make_shortcut.bat` once. It makes `FobFit.lnk` in this folder. Drag that to the desktop or the Start menu. If you move the FobFit folder, run `make_shortcut.bat` again.

## Scan the key

- Fold the blade out. A flip key must be open.
- Spray glossy black plastic and the metal blade with scanning spray. They scan badly without it.
- Scan both faces and the edges. Small holes in the scan are filled; large holes are not.
- Export STL, PLY or OBJ. Millimetres are best; the tool also accepts centimetres, metres and inches.
- Also export the raw point cloud (the points before the scanner software made a mesh) if your scanner can. Put it next to the scan and give it a name that starts like the scan's, for example `toyota yaris mesh.stl` and `toyota yaris point cloud.obj`. The tool finds it and marks the surface the scanner software invented: places with no raw point within 0.5 mm.
- Measure the key with calipers: body length (from the ring end to where the blade leaves the body), largest width, largest thickness. The tool uses these to check the scan's size.

## Use the window

The window has five steps on the left and the 3D view on the right.

**3D view.** Drag with the left button to turn the view. Use the wheel or drag with the right button to zoom. Drag with the middle button, or hold Shift and drag with the left button, to move the view. The buttons under the view switch between the key, the jacket on the key, the jacket alone and the deviation map.

**1 Scan**

- **Open scan...** Choose the scan file. The tool reads it, removes loose fragments, fills small holes and smooths scan noise. The line below shows what it did.
- **units** The tool guesses the units from the size of the key. Change this only if the guess is wrong.
- **Raw point cloud** The line shows the point cloud the tool found next to the scan. **Choose...** picks another file; **None** uses no point cloud. Without one, invented surface is not marked.

**2 Check what the tool found.** Look at the key in the 3D view:

- The **blue arrow** runs along the blade axis and points to the blade tip.
- The **green arrow** points out of the button face. The key snaps into the jacket from this side.
- A **red circle** marks the key-ring hole.

If something is wrong, correct it:

- **Flip button face** turns the key over.
- **Swap blade / ring ends** turns the key end for end.
- **Pick blade axis (2 clicks)** Click the blade tip, then the blade where it leaves the body.
- **Mark a button** Click the middle of a button. The jacket keeps a circle of the mark radius free there. Buttons on the face stay open anyway, so mark buttons on the side, such as the release button of a flip key.
- **Mark ring hole** Click the wall of the ring hole, if the tool did not find it.
- **No ring hole** Use this if the key has no ring hole.
- **Mark flip-key pivot** Click the hinge of a flip key's blade (often under its release button). The jacket then keeps the path of the folding blade free. Click the pivot on the side the blade folds to. This costs grip: inside the blade's swing the wall on that side stops below the blade, so that side has no lip there. If the blade can stay open while the jacket is on, leave the pivot unmarked; the jacket then holds better.
- **Clear marks** removes the buttons and the pivot.
- **mark radius** sets the size of the next button mark.

**3 Calipers (optional).** Type the body length, the largest width and the largest thickness in millimetres, then press **Check**. The tool compares them with the scan. Many consumer scanners scan a few tenths of a millimetre small or large. If the scan is off, the tool suggests a scale. Tick **Rescale the scan to the calipers** to use it. The tool never rescales unless you tick this box. The length is shown only: where a body ends is a judgement, so the scale uses width and thickness.

**4 Wing style**

- **Wing**: two short wings, 45 mm lever arm. For an ignition.
- **Paddle**: one long wing, 70 mm lever arm. For a stiff door lock.
- **Loop**: a ring for a hooked finger or the thumb, 40 mm from the blade axis.
- **lever arm** is the distance from the blade axis to the centre of the round tip (or of the ring hole). A longer arm needs less force, but the jacket gets bigger.
- **side** sets which side a one-sided Paddle or Loop goes on.
- **Advanced fit settings** shows four numbers. Change them only after printing a first jacket:
  - **gap**: space between key and cavity, 0.30 mm to start.
  - **lip undercut**: how far the lips reach over the widest edge of the key. 0 lets the tool choose: it takes the deepest undercut that keeps the lip within the strain limit for PETG (1.5 %) and the snap-in force under 120 N, in 0.05 mm steps. That holds the key as hard as the plastic safely allows.
  - **side wall** and **thinnest back** set the wall thicknesses.

**5 Make FobFit.** Press the button. The window stays usable and shows progress. A jacket takes about one minute. The view then shows the jacket on the key, and the box below lists the results and any warnings. **Open output folder** opens the files. **Output folder...** chooses where they go; by default they go to `out\<scan name>_<style>` in the FobFit folder, never into the scan's folder. Tick **Slice** to get sliced print files (needs Bambu Studio).

## What you get

| File | What it is |
|---|---|
| `..._FobFit_<style>.stl` | the jacket in its print pose: flat back on the bed, opening up, no supports |
| `..._FobFit_<style>.3mf` | the same, sliced in PETG for a Bambu Lab printer if you ticked slicing (open it in Bambu Studio and print) |
| `..._report.md` / `.json` | mass, size, gap statistics, invented surface, lips, wing, the checks, the warnings, and the run time and peak memory of each stage |
| `..._deviation_map.png` | the scan coloured by its gap to the cavity, against a cavity made from three caliper sizes; invented surface in purple |
| `..._render.png`, `..._section.png` | pictures of the jacket on the key and a cross-section through the lips |
| `..._scan_in_print_pose.stl` | the scan where it sits in the printed jacket, for checking in a slicer or viewer |
| `..._FobFit_settings.json` | every setting of this run, to repeat it from the command line |

## Print and fit

1. Print the jacket (the report gives its print time).
2. To fit it, lay the jacket on a table, opening up, and press the key in with the palm, back first.
3. If the key rattles, make the jacket again with a smaller **gap** under **Advanced fit settings** (0.30 mm to start; try 0.20). If it does not snap in, try 0.40.

Do not force a stuck lock. A long lever can pass the torque a key is built to survive.

## Command line

The same tool runs without the window:

```bash
python fobfit_cli.py key_scan.stl --style paddle --calipers 62.0,36.1,16.2 --apply-scale --slice
```

`python fobfit_cli.py --help` lists every option. Positions for `--button`, `--ring` and `--fold` are in the key frame the report gives: x along the blade toward the tip, z out of the button face, origin on the blade axis at the middle of the body. `--settings` repeats a window run from its settings file. `--cloud` names the raw point cloud (`--cloud none` uses none). `--out` names the output folder.

If a fit surface depends on invented surface, the report warns: over 5 % of a lip, or over 10 % of the blade-end wall (the last 4 mm of the body). Gap figures in the report count measured surface only.

## Tests

```bash
python tests/run_tests.py
python fobfit_gui.py --demo tests/gui/demo_wedge.json --shots tests/gui/shots_wedge
```

The first runs the generator on synthetic keys, clean and degraded (noise, holes, loose fragments, any pose, a 0.5 % scale error, metres), and checks each jacket against the true key. The second drives the window from a script and saves screenshots.

## How it works

The tool puts the scan in a key frame (blade axis, button face), builds a signed distance field of the key on a 0.2 mm grid and derives the jacket from it. The cavity is the key grown by the gap, plus everything the key passes through when it snaps in from the button side, so nothing blocks it on the way in except the lips. The lips reach past the widest contour only along the long sides, away from the ends and corners, where the walls can flex. The outside has a flat back for printing, a rim that follows the key's edge, and a wing in the bed plane, square to the blade axis. The code is in `fobfit/`; the wing styles in `fobfit/wings.py` do not depend on the scan, so a jacket made from caliper sizes can reuse them. `BUILD_RECORD.md` holds the design calculations and the test results.
