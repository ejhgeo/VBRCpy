# Why MAP and Mean Behave Differently in the Round-Trip vs PREM Validation

## The T–φ Trade-Off Ridge

At every depth, the posterior probability has weight spread along a **T–φ trade-off ridge**: many (T, φ) combinations produce similar Vs. A hotter, melt-free mantle looks the same as a colder mantle with some melt.

- **Joint MAP** picks the single voxel with the highest posterior probability.
- **Marginal mean** for T integrates (averages) over all φ and gs values, weighted by the posterior.

The mean is *always* pulled cold by the ridge, in both cases, because there are many (cold-T, nonzero-φ) voxels with significant posterior weight that drag the weighted average downward. This is a volume effect — there's more posterior "stuff" on the cold side of the ridge than at the peak.

## Why They *Look* Different

### Round-trip case

The synthetic Vs was generated from a **single exact grid point** (e.g., T=1370, φ=0, gs=10000). The likelihood is maximally peaked at that point — it's the one (T, φ, gs) combination that reproduces the observed Vs perfectly. So the posterior has a **sharp, dominant peak** at the true values.

- **MAP** finds that peak reliably at every depth → smooth, correct profile.
- **Mean** still sees all the (cold-T, nonzero-φ) probability mass on the ridge and gets pulled cold → biased, and the bias varies slightly depth-to-depth → noisier.

### PREM case

The observed Vs came from the real Earth, not from one point on our grid. No single (T, φ, gs) voxel perfectly "owns" the data. The posterior ridge is **broad and flat** — many combos fit nearly equally well.

- **MAP**: Whichever voxel happens to have marginally higher probability wins, but this winner changes erratically from depth to depth (hot+dry at one depth, cold+melty at the next) → noisy profile.
- **Mean**: By averaging over the flat ridge, produces a stable answer at every depth → smooth profile. It's still biased cold (same mechanism as round-trip), but you can't see the bias because there's no known "true T" to compare against.

## Summary Table

|                | Posterior shape                      | MAP behavior               | Mean behavior                     |
|----------------|--------------------------------------|----------------------------|-----------------------------------|
| **Round-trip** | Sharp peak (data from one grid point)| Stable, correct            | Smooth-ish but biased cold        |
| **PREM**       | Flat ridge (data from nature)        | Noisy (jumping along ridge)| Smooth but biased cold            |

The mean is *always* biased cold. The MAP is *always* trying to find the peak. The difference is whether that peak is sharp (round-trip → MAP wins) or flat (PREM → MAP is noisy, mean's smoothness is more useful despite the bias).
