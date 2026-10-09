# GODUNOV_SPH_PLAN L1b: the papers' state limiters (2026-10-08)

`StateLimiter` (`modules/reconstruction/pairState.py`): the second-order states of the pair Riemann problem limited along the pair axis `s` from
the finite difference `D = Q_i - Q_j` and each particle's projected SPH gradient `Delta = g . x_ij` (applied to `rho`, `P` and the normal velocity):

| config | limiter | source |
|---|---|---|
| gsphO2 | `PairRatio`: the AV_PLAN Phase 3 ratio-of-gradients limiter, states clamped between the particle values | what layers 1-2 used |
| gsphO2H | `VanLeerHarmonic`: `2 D Delta / (D + Delta)` if `D Delta > 0` else 0 | Murante et al. 2011 Eqs. 18-23 (their reference) |
| gsphO2M | `VanLeerMonotonized`: `sgn min(2\|D\|, \|(D + Delta)/2\|, 2\|Delta\|)` if `D Delta > 0` else 0 | Iwasaki & Inutsuka 2011 App. C / van Leer 1979 |
| gsphO2I | `InutsukaSign`: unlimited gradients, both zero where the two disagree in sign | Inutsuka 2002 Eq. 74 |
| gsphO2HS / IS | the above with the first-order shock switch, `C (v_j - v_i) . n > min(c_i, c_j)`, `C = 3` | Inutsuka 2002 Eq. 75 |

All on the simplified Godunov SPH (layer 2), second-order velocity and `(rho, P)`, Adaptive solver; full profile, video, no velocity alarm.

| case metric | gsphO2 | gsphO2H | gsphO2M | gsphO2I | gsphO2HS | gsphO2IS |
|---|---|---|---|---|---|---|
| Sod 1D  L1(v_x) | **0.00286** | 0.00439 | 0.00483 | 0.00449 | 0.00457 | 0.00448 |
| Sod 1D  P contact spike | 0.0104 | 0.0090 | 0.0103 | 0.0106 | 0.0204 | 0.0111 |
| Sod 1D  entropy contact spike | 0.085 | 0.081 | 0.082 | **0.053** | 0.114 | 0.071 |
| Sod 1D  energy drift | 5.4e-5 | 3.7e-5 | 3.4e-5 | 1.7e-5 | 3.7e-5 | 1.7e-5 |
| Sod 2D  L1(v_x) | 0.0167 | **0.0155** | 0.0177 | 0.0174 | 0.0191 | 0.0218 |
| Noh  post-shock rho error | -0.0007 | -0.0032 | -0.0037 | -0.0016 | -0.0005 | **-0.0005** |
| Gresho  L1(v_phi) | 0.129 | 0.109 | **0.0958** | 0.119 | 0.109 | 0.119 |
| Gresho  peak speed (~0.7) | 0.60 | 0.66 | **0.69** | 0.62 | 0.66 | 0.62 |
| Sedov  peak rho ratio (exact 4) | 2.41 | 2.65 | **2.80** | 2.54 | 2.22 | 2.23 |
| Sedov  shock radius error | -0.048 | -0.043 | -0.046 | -0.054 | **-0.030** | -0.046 |
| KH (nx 128)  A(t = 1.5), reference 0.148 | 0.0898 | 0.1044 | **0.1056** | 0.0924 | 0.1044 | 0.0924 |

## Reading

- **The papers' limiters beat the AV-derived one where the flow is smooth or the structures are fine**: Gresho L1 0.129 -> 0.096 (M) / 0.109 (H), the KH amplitude 0.090 -> 0.104-0.106 (the AV default
  0.116, the best limited-AV rows 0.098-0.100), the Sedov peak density 2.41 -> 2.65-2.80, energy drift 1.5-3x lower. Sod 1D is slightly worse (0.0029 -> 0.0044) and the Noh post-shock density error larger
  (-0.0007 -> -0.003).
- **The first-order shock switch fixes that Noh error** (-0.0005) and the Sedov radius (-0.030, the best of any variant) at the price of a doubled Sod pressure / entropy spike.
- **Murante's van Leer limiters (H, M) are the better default** than the ratio limiter on the structure tests; `VanLeerMonotonized` is best on Gresho / Sedov / KH, `VanLeerHarmonic` is Murante's reference and
  better on Sod 2D. `InutsukaSign` has the smallest entropy blip and energy drift. Recommendation: `VanLeerHarmonic` or `VanLeerMonotonized` as the default, shock switch on if shocks matter more than the contact;
  not changed here (the `GSPHConfig` default is still `PairRatio`): the choice is the user's.
- Checks: `tests/test_pairState.py` (1D limiters exact for linear fields, step / extremum -> first order, boundedness and `i <-> j` symmetry), `scripts/gradcheck_godunov.py`.
