# Calculating Total Energy with Plane Waves

In this tutorial, we derive the total energy functional using a **Plane Wave Basis Set** within the framework of **Density Functional Theory (DFT)**. This approach is widely used in computational materials science due to its efficiency for periodic systems.


## The Plane Wave Basis Set

For a periodic system, Bloch’s theorem allows us to express the wavefunction as a product of a plane wave and a periodic function $u_{i, \boldsymbol{k}}  (\boldsymbol{r})$. Expanding $u_{i, \boldsymbol{k}}  (\boldsymbol{r})$ in the plane wave basis set, we have

$$
u_{i, \boldsymbol{k}}  (\boldsymbol{r}) = \dfrac{1}{  \sqrt{\Omega_\text{cell}}}    \sum_{\boldsymbol{G}} c_{i, \boldsymbol{k}, \boldsymbol{G} }	  \exp(\text{i} \boldsymbol{G}\cdot \boldsymbol{r} )
$$

where
- $\boldsymbol{r}$: Position vector in real space;
- $i$: Band index;
- $\boldsymbol{k}$: Wavevector in the Brillouin zone;
- $\boldsymbol{G}$: Reciprocal lattice vector (indexes plane waves); 
- $c_{i, \boldsymbol{k}, \boldsymbol{G} }$: Expansion coefficients (complex numbers);
- $\Omega_\text{cell}$: Volume of the unit cell.

The complete wave function can be then written as

$$
\psi_{i, \boldsymbol{k}}  (\boldsymbol{r}) = \exp(\text{i} \boldsymbol{k}\cdot \boldsymbol{r} ) u_{i, \boldsymbol{k}}  (\boldsymbol{r}).
$$

### Unitarity Constraint
Density Functional Theory (DFT) employs single-particle wave functions under the assumption of orthonormality. This requirement translates to a constraint on the plane wave expansion coefficients $c_{i, \boldsymbol{k}, \boldsymbol{G}}$, which must satisfy the orthogonality condition at eacheach $k$-point:

$$
\sum_{\boldsymbol{G}} c_{i, \boldsymbol{k}, \boldsymbol{G}}^* c_{j, \boldsymbol{k}, \boldsymbol{G}} = \delta_{ij}, \quad \forall \ \boldsymbol{k}.
$$

In our implementation, the orthonormality of the wave functions within the plane wave basis set is enforced via a QR decomposition. This is performed using the `jax.numpy.linalg.qr` function, which employs the Householder transformation—a numerically stable method for orthogonalization.


## Total Energy as the Objective Function
The objective is to minimize the total energy of the system with respect to the plane wave coefficients.

$$
\min_{\psi_{i, \boldsymbol{k}}} E_{\text{total}}[\left\{ \psi_{i, \boldsymbol{k}} \right\}]
$$

where the $E_{\text{total}}$ is the total energy functional of wave functions. The total energy functional in density functional theory (DFT) is constructed by the kinetic energy functional $E_{\text{kinetic}}$, the electron-ion interaction energy functional, or the external potential energy functional $E_{\text{external}}$, the electron-electron interaction energy functional, also know as the Hartree energy functional $E_{\text{hartree}}$, and the exchange-correlation energy functional $E_{\text{xc}}$, and the nuclear-nuclear interaction energy functional $E_{\text{nuclear}}$:

$$
E_{\text{total}} = E_{\text{kinetic}} + E_{\text{external}} + E_{\text{hartree}} + E_{\text{xc}} + E_{\text{nuclear}}.
$$


### The Kinetic Energy Functional
The kinetic energy functional is given by

$$
\begin{aligned}
E_{\text{kinetic}} &= \dfrac{1}{2} \sum_{i, \boldsymbol{k}} \int_{\Omega_\text{cell}} \psi_{i, \boldsymbol{k}} (\boldsymbol{r})  \nabla_{\boldsymbol{r}} \psi_{i, \boldsymbol{k}} (\boldsymbol{r}) d\boldsymbol{r}.  \\
&= \dfrac{1}{2} \sum_{i, \boldsymbol{k}} \sum_{\boldsymbol{G}} \left\Vert \boldsymbol{k}+\boldsymbol{G} \right\Vert^2 \left\Vert c_{i, \boldsymbol{k}, \boldsymbol{G}} \right\Vert^2.  \\
\end{aligned}
$$

### The External Potential Energy Functional
The external potential energy functional is given by

$$
E_{\text{external}} =  \\
$$
