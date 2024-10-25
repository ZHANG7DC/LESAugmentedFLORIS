
## Toward Ultra-Efficient High-Fidelity Predictions of Wind Turbine Wakes

### Overview

This repository contains the code accompanying the paper:

**Toward ultra-efficient high-fidelity predictions of wind turbine wakes: Augmenting the accuracy of engineering models with machine learning**  
*Santoni, C., Zhang, D. (张迪畅), Zhang, Z. (张泽夏), Samaras, D., Sotiropoulos, F., and Khosronejad, A.*  
Published in: *Physics of Fluids*, Vol. 36, Issue 6, 2024  
[https://doi.org/10.1063/5.0213321](https://doi.org/10.1063/5.0213321)

---

### Abstract

This study proposes a novel machine learning (ML) methodology for the efficient and cost-effective prediction of high-fidelity three-dimensional velocity fields in the wake of utility-scale turbines. The model consists of an autoencoder convolutional neural network with U-Net skipped connections, fine-tuned using high-fidelity data from large-eddy simulations (LES). 

The trained model takes the low-fidelity velocity field cost-effectively generated from the analytical engineering wake model as input and produces high-fidelity velocity fields. The ML model reduces the prediction error from the 20% obtained by the Gauss Curl Hybrid (GCH) model to less than 5%. It also accurately captures non-symmetric wake deflection for opposing yaw angles, outperforming the GCH model. Additionally, the computational cost of the ML model is comparable to that of the analytical wake model while providing results nearly as accurate as those from LES.

---

### Citation

If you use this code, please cite the following paper:

```bibtex
@article{10.1063/5.0213321,
    author = {Santoni, C. and Zhang, D. (张迪畅) and Zhang, Z. (张泽夏) and Samaras, D. and Sotiropoulos, F. and Khosronejad, A.},
    title = "{Toward ultra-efficient high-fidelity predictions of wind turbine wakes: Augmenting the accuracy of engineering models with machine learning}",
    journal = {Physics of Fluids},
    volume = {36},
    number = {6},
    pages = {065159},
    year = {2024},
    month = {06},
    doi = {10.1063/5.0213321},
    url = {https://doi.org/10.1063/5.0213321},
    eprint = {https://pubs.aip.org/aip/pof/article-pdf/doi/10.1063/5.0213321/20008573/065159_1_5.0213321.pdf}
}
```

---



### License

This code is distributed for research purposes only. Please contact the authors if you wish to use it in commercial applications.

---

### Contact

For questions or issues, please contact **D. Zhang** at diczhang@cs.stonybrook.edu.

---
