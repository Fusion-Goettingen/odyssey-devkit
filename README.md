<div align="center">
    <h1>Odyssey</h1>
    <br />
    <br />
    <a href="https://arxiv.org/abs/2512.14428">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/Fusion-Goettingen/odyssey-devkit#pencil-citation">Citation</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/Fusion-Goettingen/odyssey-devkit#mailbox-contact">Contact</a>
  <br />
  <br />
</p>
</div>

![Titleimage](titleimage.jpg)
Odyssey is an automotive dataset designed for LiDAR and LiDAR-inertial odometry, as well as other localization tasks such as place recognition. This repository contains the accompanying code, including a Python data loader and examples demonstrating its use.

## :confetti_ball: Odyssey accepted at IJRR :confetti_ball:
We are happy to announce that **Odyssey** has been accepted for publication in *The International Journal of Robotics Research (IJRR)*.

## Python Dependencies
- NumPy
- SciPy
- Matplotlib (required for `example.py` only)

## Quickstart
Clone this repo with
```bash
git clone git@github.com:Fusion-Goettingen/odyssey-devkit.git
cd odyssey-devkit
```
modify the `base_dir` and `seq` in `example.py` to point to the directory of the Odyssey dataset and execute with
```bash
python3 example.py
```
to see our dataloader in action.

## :pencil: Citation
Odyssey has been accepted for publication in *The International Journal of Robotics Research (IJRR)* and is currently in production. Until the final published version becomes available, please cite the arXiv preprint below. We will update this citation once the final version is available.

```
@ARTICLE{kurda26odyssey,
  author={Aaron Kurda and Simon Steuernagel and Lukas Jung and Marcus Baum},
  year={2026},
  eprint={2512.14428},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.14428}, 
}
```

## :bug: Found a Bug ?
Feel free to open an [Issue](https://github.com/Fusion-Goettingen/odyssey-devkit/issues)

## :mailbox: Contact

Aaron Kurda :envelope: aaron.kurda@cs.uni-goettingen.de