# Citation verification log

Every reference used in `docs/report.md` was checked on 2026-11-04 by fetching an official page from
inside the development VM. The VM's network policy only allowed GitHub (arxiv.org, dl.acm.org,
ieeexplore.ieee.org, ojs.aaai.org, sciencedirect.com and readthedocs.io did not resolve), so
verification used the projects' official GitHub repositories and the citation blocks in their READMEs.
References that could not be verified against an official page are **not cited** in the report; they
are listed at the bottom for completeness.

| Key | Reference as cited | Official page fetched | Status |
|---|---|---|---|
| sumo-rl | Alegre, L. N. *SUMO-RL* (2019), GitHub repository, https://github.com/LucasAlegre/sumo-rl | https://github.com/LucasAlegre/sumo-rl (README citation block, BibTeX "SUMO-RL", year 2019) | verified |
| sb3 | Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N. *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine Learning Research 22(268):1-8, 2021 | https://github.com/DLR-RM/stable-baselines3 (README "Citing the Project") | verified |
| sb3-contrib / MaskablePPO | Stable-Baselines3 Contrib, "PPO with invalid action masking (MaskablePPO)", https://github.com/Stable-Baselines-Team/stable-baselines3-contrib; the README links the masking method to arXiv:2006.14171 | https://github.com/Stable-Baselines-Team/stable-baselines3-contrib (README feature list) | verified (repository and arXiv id); the arXiv page itself could not be fetched, so the paper's title/authors are not stated in the report |
| presslight | Wei, H., Chen, C., Zheng, G., Wu, K., Gayah, V., Xu, K., Li, Z. *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*. KDD '19, pp. 1290-1298, 2019 | https://github.com/wingsweihua/presslight (README citation) | verified |
| colight | Wei, H., Xu, N., Zhang, H., Zheng, G., Zang, X., Chen, C., Zhang, W., Zhu, Y., Xu, K., Li, Z. *CoLight: Learning Network-level Cooperation for Traffic Signal Control*. CIKM '19, 2019 | https://github.com/wingsweihua/colight (README citation) | verified |
| sumo | Eclipse SUMO, Simulation of Urban MObility, https://github.com/eclipse-sumo/sumo (Zenodo DOI badge 10.5281/zenodo.23219601) | https://github.com/eclipse-sumo/sumo | verified as software citation; the ITSC 2018 paper page (ieeexplore) could not be fetched and is not cited |
| baselines | Dhariwal, P. et al. *OpenAI Baselines* (2017), GitHub repository, https://github.com/openai/baselines (PPO2 implementation) | https://github.com/openai/baselines (README citation) | verified |

## Candidates that could not be verified (not cited in the report)

- Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347): arxiv.org unreachable. PPO is therefore attributed in the report to the Stable-Baselines3 implementation and the OpenAI Baselines repository, both verified above.
- Varaiya, *Max pressure control of a network of signalized intersections* (Transportation Research Part C, 2013): sciencedirect.com unreachable. The report describes max-pressure control as "the max-pressure rule as implemented in PressLight's baseline" and cites PressLight, which is verified.
- Chen et al., *MPLight* (AAAI 2020): the official repository README (https://github.com/Chacha-Chen/MPLight) does not name the paper and ojs.aaai.org was unreachable; not cited.
- Alshiekh et al., *Safe Reinforcement Learning via Shielding* (AAAI 2018) and Huang & Ontañón, *A Closer Look at Invalid Action Masking* (arXiv:2006.14171): pages unreachable; the shield and masking are described as this project's own mechanisms, with the masking implementation credited to sb3-contrib (verified).
- Webster, *Traffic Signal Settings* (Road Research Technical Paper 39, 1958): no official online page reachable; the fixed-time plan is described as "a Webster-style cycle calculation" without a formal citation.
