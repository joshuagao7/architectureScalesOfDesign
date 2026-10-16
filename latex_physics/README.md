# Double Pendulum Mathematics

This folder contains LaTeX documentation for the mathematical foundations of the double pendulum system.

## Contents

- `double_pendulum_math.tex` - Complete LaTeX document covering:
  - Lagrangian mechanics derivation
  - Euler-Lagrange equations of motion
  - Numerical methods (RK4 integration)
  - Inverse kinematics
  - Chaos theory and Lyapunov exponents
  - Phase space analysis
  - Small angle approximation and normal modes

## Compiling

To compile the PDF, run:

```bash
pdflatex double_pendulum_math.tex
pdflatex double_pendulum_math.tex  # Run twice for table of contents
```

Or using latexmk:

```bash
latexmk -pdf double_pendulum_math.tex
```

## Dependencies

The LaTeX document requires the following packages:
- amsmath, amssymb, amsthm
- geometry
- graphicx
- tikz (with angles, quotes, calc libraries)
- hyperref
- fancyhdr
- xcolor

These are included in most standard LaTeX distributions (TeX Live, MiKTeX, TinyTeX).
