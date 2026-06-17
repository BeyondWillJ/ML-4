# Track Finding and Momentum Prediction

## File Structure

- Data files are stored in the `data` directory.
- Function libraries: `SFlib_I.py`, `SFlib_II.py`
- Code files: `FindTrack.py`, `Predict.py`

## Track Finding

By analyzing particle collision event data (including noise data), identify and reconstruct particle track classifications, and display them in figures.

- Directly runnable script file: `FindTrack.py`
- Data file used: `data/SiHits_3D_pvar_zvar_0.03_0.50_28_0.04_0.06_2000_v1.txt`

In addition, the author has created a corresponding UI interface to facilitate users to interactively select events and draw tracks. Simply run `FindTrack_UI.py` to use it.

## Momentum Prediction

Extract features such as curvature/center and fitting errors of tracks in the x–y plane through geometric clustering and circle fitting, train an MLP regressor to predict the transverse momentum components `(p_x, p_y)` of particles, and evaluate the deviation distribution between predictions and true values.

- Directly runnable script file: `Predict.py`
- Data file used: `data/SiHits_3D_pvar_0.02_10000_v2.txt`
- Two images are generated: `figure_predict_MSE.png` and `figure_predict_partII.png`, which show the prediction effect of the model on the test set and the histogram of evaluation results, respectively.

## References

This project is an implementation of the following literature: `J. Zhi, S. Wu, J. Zhao and X. Cao, "Hybrid Algorithms for Enhanced Vertex and Track Reconstruction," 2024 6th International Communication Engineering and Cloud Computing Conference (CECCC), Chengdu, China, 2024, pp. 15-22, doi: 10.1109/CECCC62598.2024.11063558.`

url: https://ieeexplore.ieee.org/abstract/document/11063558
