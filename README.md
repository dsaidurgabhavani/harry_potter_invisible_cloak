# 🪄 Harry Potter Invisibility Cloak with OpenCV & Python

Make any covered object or person invisible in real time using computer vision! When you wrap an object or yourself in a cloth, the program replaces that cloth area with the previously captured background, creating the illusion of transparency.

---

## 📌 How It Works (The Core Concept)

The invisibility effect works through **chroma-keying / color substitution** in 5 steps:

```
[Webcam Starts]
      │
      ▼
1. Capture Static Background (Empty room before you step in)
      │
      ▼
2. Read Video Feed & Convert Frame from BGR to HSV Color Space
      │
      ▼
3. Threshold Cloth Color -> Generate Binary Mask
      │
      ▼
4. Morphological Filtering (Remove noise + Smooth edges)
      │
      ▼
5. Blend Background (Cloak Area) + Current Frame (Non-Cloak Area)
      │
      ▼
[Invisibility Illusion Output]
```

### Why HSV instead of RGB / BGR?
In RGB/BGR color space, pixel values are highly sensitive to shadows and brightness changes. **HSV** (Hue, Saturation, Value) separates color pigment (**Hue**) from illumination (**Value**), making color detection much more reliable under room lighting.

---

## 🚀 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.8+ installed, then install OpenCV and NumPy:
```bash
pip install -r requirements.txt
```
*(or manually: `pip install opencv-python numpy`)*

### 2. Run the Application

#### Option A: Feature-Rich Interactive Version (Recommended)
```bash
python invisibility_cloak.py
```
**Features included:**
- On-screen 3-second countdown for background capture
- Instant background recapture anytime by pressing `b`
- Color presets: **Red** (`1`), **Green** (`2`), **Blue** (`3`)
- **Click-to-calibrate** (`c`): Click on your cloth on-screen to automatically detect and track that exact fabric!
- **Real-time HSV sliders** (`t`): Open trackbars to fine-tune lower & upper bounds live
- **Debug split view** (`m`): See original feed, binary mask, and final output side-by-side
- **Save snapshots** (`s`)

#### Option B: Minimal Beginner Version
```bash
python simple_cloak.py
```
A clean, minimal ~50-line script ideal for presentations, school projects, or understanding the raw computer vision algorithm.

---

## ⌨️ Interactive Controls (`invisibility_cloak.py`)

| Key | Action |
| :--- | :--- |
| `b` / `B` | **Recapture Background** (useful if camera moved or lighting changed) |
| `1` | Switch to **Red** cloth preset |
| `2` | Switch to **Green** cloth preset |
| `3` | Switch to **Blue** cloth preset |
| `c` / `C` | **Click-to-calibrate** — Click directly on your cloth in the window |
| `t` / `T` | Toggle **HSV trackbars** for manual fine-tuning |
| `m` / `M` | Toggle **Debug View** (Original \| Mask \| Result) |
| `s` / `S` | Save a snapshot image |
| `q` / `Esc` | Quit the application |

---

## 💡 Tips for Best Results

1. **Keep the Camera Fixed**: Mount your camera or keep your laptop steady on a desk. If the camera shifts, the background frame will misalign with the live view (press `b` to recapture if this happens).
2. **Lighting**: Uniform ambient lighting works best. Avoid heavy backlighting or strong harsh spotlights that cast dark shadows onto the cloth.
3. **Cloth Color Choice**:
   - Use a solid, non-reflective cloth (bright red, royal blue, or vibrant green).
   - Ensure the clothes you are wearing underneath do **not** match the color of the cloak (otherwise those parts will turn invisible too!).
   - Ensure your background does **not** have objects of the exact same color as your cloak.
4. **Fine-Tuning**: If parts of the cloth are not turning invisible or fringes are visible, press `t` to adjust the HSV sliders or press `c` and click on the cloth to calibrate.
